"""Dense OSQP-style ADMM kernels exposed through a stable C ABI."""

from std.math import sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime W = simdwidthof[DType.float64]()
comptime MAX_DIMENSION = 1000000000
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def ptr(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


def valid_address(address: Int, length: Int) -> Bool:
    return length == 0 or address != 0


def valid_dimensions(n: Int, m: Int) -> Bool:
    return n >= 0 and m >= 0 and n <= MAX_DIMENSION and m <= MAX_DIMENSION


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var accum = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        accum += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = accum.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def matvec(a: Ptr, x: Ptr, result: Ptr, rows: Int, cols: Int):
    for row in range(rows):
        result[row] = dot(a + row * cols, x, cols)


def diagonal_matvec(a: Ptr, x: Ptr, result: Ptr, n: Int):
    var i = 0
    while i + W <= n:
        result.store(
            i,
            a.load[width=W](i) * x.load[width=W](i),
        )
        i += W
    while i < n:
        result[i] = a[i] * x[i]
        i += 1


def transposed_matvec(a: Ptr, x: Ptr, result: Ptr, rows: Int, cols: Int):
    var col = 0
    while col + W <= cols:
        result.store(col, SIMD[DType.float64, W](0.0))
        col += W
    while col < cols:
        result[col] = 0.0
        col += 1

    for row in range(rows):
        var arow = a + row * cols
        var value = SIMD[DType.float64, W](x[row])
        col = 0
        while col + W <= cols:
            result.store(
                col,
                result.load[width=W](col) + arow.load[width=W](col) * value,
            )
            col += W
        while col < cols:
            result[col] += arow[col] * x[row]
            col += 1


def symmetric_matvec(a: Ptr, x: Ptr, result: Ptr, n: Int):
    for row in range(n):
        result[row] = dot(a + row * n, x, n)


def factor_system(
    p: Ptr,
    a: Ptr,
    rho: Ptr,
    factor: Ptr,
    n: Int,
    m: Int,
    sigma: Float64,
    diagonal: Bool,
) -> Bool:
    if diagonal:
        var i = 0
        var sigma_vec = SIMD[DType.float64, W](sigma)
        while i + W <= n:
            var a_vec = a.load[width=W](i)
            factor.store(
                i,
                p.load[width=W](i)
                + sigma_vec
                + rho.load[width=W](i) * a_vec * a_vec,
            )
            i += W
        while i < n:
            factor[i] = p[i] + sigma + rho[i] * a[i] * a[i]
            i += 1
        for j in range(n):
            if factor[j] <= 0.0:
                return False
        return True

    for row in range(n):
        var factor_row = factor + row * n
        var p_row = p + row * n
        var col = 0
        while col + W <= row + 1:
            factor_row.store(col, p_row.load[width=W](col))
            col += W
        while col < row + 1:
            factor_row[col] = p_row[col]
            col += 1
        factor[row * n + row] += sigma

    for constraint in range(m):
        var weight = rho[constraint]
        var arow = a + constraint * n
        for row in range(n):
            var scaled = weight * arow[row]
            if scaled != 0.0:
                var factor_row = factor + row * n
                var scaled_vec = SIMD[DType.float64, W](scaled)
                var col = 0
                while col + W <= row + 1:
                    factor_row.store(
                        col,
                        factor_row.load[width=W](col)
                        + scaled_vec * arow.load[width=W](col),
                    )
                    col += W
                while col < row + 1:
                    factor_row[col] += scaled * arow[col]
                    col += 1

    for row in range(n):
        var factor_row = factor + row * n
        for col in range(row + 1):
            var value = factor_row[col] - dot(
                factor_row, factor + col * n, col
            )
            if row == col:
                if value <= 0.0:
                    return False
                factor_row[col] = sqrt(value)
            else:
                factor_row[col] = value / factor[col * n + col]
    return True


def solve_factor(factor: Ptr, rhs: Ptr, n: Int):
    for row in range(n):
        var value = rhs[row] - dot(factor + row * n, rhs, row)
        rhs[row] = value / factor[row * n + row]

    for reverse_row in range(n):
        var row = n - 1 - reverse_row
        rhs[row] /= factor[row * n + row]
        var factor_row = factor + row * n
        var value = SIMD[DType.float64, W](rhs[row])
        var col = 0
        while col + W <= row:
            rhs.store(
                col,
                rhs.load[width=W](col)
                - factor_row.load[width=W](col) * value,
            )
            col += W
        while col < row:
            rhs[col] -= factor_row[col] * rhs[row]
            col += 1


def build_rhs(
    a: Ptr,
    q: Ptr,
    rho: Ptr,
    x: Ptr,
    y: Ptr,
    z: Ptr,
    rhs: Ptr,
    n: Int,
    m: Int,
    sigma: Float64,
):
    var col = 0
    var sigma_vec = SIMD[DType.float64, W](sigma)
    while col + W <= n:
        rhs.store(
            col,
            sigma_vec * x.load[width=W](col) - q.load[width=W](col),
        )
        col += W
    while col < n:
        rhs[col] = sigma * x[col] - q[col]
        col += 1

    for row in range(m):
        var value = rho[row] * z[row] - y[row]
        var value_vec = SIMD[DType.float64, W](value)
        var arow = a + row * n
        col = 0
        while col + W <= n:
            rhs.store(
                col,
                rhs.load[width=W](col)
                + arow.load[width=W](col) * value_vec,
            )
            col += W
        while col < n:
            rhs[col] += arow[col] * value
            col += 1


def infinity_norm(values: Ptr, n: Int) -> Float64:
    var largest = 0.0
    for i in range(n):
        var value = abs(values[i])
        if value > largest:
            largest = value
    return largest


def compute_residuals(
    p: Ptr,
    a: Ptr,
    q: Ptr,
    z: Ptr,
    x: Ptr,
    y: Ptr,
    ax: Ptr,
    px: Ptr,
    aty: Ptr,
    n: Int,
    m: Int,
    diagonal: Bool,
) -> Tuple[Float64, Float64, Float64, Float64]:
    if diagonal:
        diagonal_matvec(a, x, ax, n)
        diagonal_matvec(p, x, px, n)
        diagonal_matvec(a, y, aty, n)
    else:
        matvec(a, x, ax, m, n)
        symmetric_matvec(p, x, px, n)
        transposed_matvec(a, y, aty, m, n)

    var primal = 0.0
    for i in range(m):
        var value = abs(ax[i] - z[i])
        if value > primal:
            primal = value

    var dual = 0.0
    for i in range(n):
        var value = abs(px[i] + q[i] + aty[i])
        if value > dual:
            dual = value

    var primal_scale = max(infinity_norm(ax, m), infinity_norm(z, m))
    var dual_scale = max(infinity_norm(px, n), infinity_norm(aty, n))
    dual_scale = max(dual_scale, infinity_norm(q, n))
    return primal, dual, primal_scale, dual_scale


@export("mosqp_factor")
def mosqp_factor(
    p_address: Int,
    a_address: Int,
    rho_address: Int,
    factor_address: Int,
    n: Int,
    m: Int,
    diagonal: Int,
    p_length: Int,
    a_length: Int,
    rho_length: Int,
    factor_length: Int,
    sigma: Float64,
) abi("C") -> Int:
    if not valid_dimensions(n, m):
        return -1
    var expected_p = n if diagonal != 0 else n * n
    var expected_a = n if diagonal != 0 else m * n
    var expected_factor = n if diagonal != 0 else n * n
    if (
        p_length != expected_p
        or a_length != expected_a
        or rho_length != m
        or factor_length != expected_factor
        or not valid_address(p_address, p_length)
        or not valid_address(a_address, a_length)
        or not valid_address(rho_address, rho_length)
        or not valid_address(factor_address, factor_length)
    ):
        return -1
    return 1 if factor_system(
        ptr(p_address),
        ptr(a_address),
        ptr(rho_address),
        ptr(factor_address),
        n,
        m,
        sigma,
        diagonal != 0,
    ) else 0


@export("mosqp_matvec")
def mosqp_matvec(
    a_address: Int,
    x_address: Int,
    result_address: Int,
    rows: Int,
    cols: Int,
    a_length: Int,
    x_length: Int,
    result_length: Int,
) abi("C") -> Int:
    if (
        not valid_dimensions(cols, rows)
        or a_length != rows * cols
        or x_length != cols
        or result_length != rows
        or not valid_address(a_address, a_length)
        or not valid_address(x_address, x_length)
        or not valid_address(result_address, result_length)
    ):
        return -1
    matvec(ptr(a_address), ptr(x_address), ptr(result_address), rows, cols)
    return 0


@export("mosqp_solve")
def mosqp_solve(
    p_address: Int,
    a_address: Int,
    q_address: Int,
    lower_address: Int,
    upper_address: Int,
    rho_address: Int,
    factor_address: Int,
    x_address: Int,
    y_address: Int,
    z_address: Int,
    ax_address: Int,
    px_address: Int,
    aty_address: Int,
    rhs_address: Int,
    info_address: Int,
    n: Int,
    m: Int,
    max_iter: Int,
    check_termination: Int,
    adaptive_interval: Int,
    diagonal_mode: Int,
    p_length: Int,
    a_length: Int,
    q_length: Int,
    lower_length: Int,
    upper_length: Int,
    rho_length: Int,
    factor_length: Int,
    x_length: Int,
    y_length: Int,
    z_length: Int,
    ax_length: Int,
    px_length: Int,
    aty_length: Int,
    rhs_length: Int,
    info_length: Int,
    eps_abs: Float64,
    eps_rel: Float64,
    alpha: Float64,
    sigma: Float64,
    adaptive_tolerance: Float64,
) abi("C") -> Int:
    if not valid_dimensions(n, m):
        return -1
    var diagonal = diagonal_mode != 0
    var expected_p = n if diagonal else n * n
    var expected_a = n if diagonal else m * n
    var expected_factor = n if diagonal else n * n
    if (
        p_length != expected_p
        or a_length != expected_a
        or q_length != n
        or lower_length != m
        or upper_length != m
        or rho_length != m
        or factor_length != expected_factor
        or x_length != n
        or y_length != m
        or z_length != m
        or ax_length != m
        or px_length != n
        or aty_length != n
        or rhs_length != n
        or info_length != 6
        or not valid_address(p_address, p_length)
        or not valid_address(a_address, a_length)
        or not valid_address(q_address, q_length)
        or not valid_address(lower_address, lower_length)
        or not valid_address(upper_address, upper_length)
        or not valid_address(rho_address, rho_length)
        or not valid_address(factor_address, factor_length)
        or not valid_address(x_address, x_length)
        or not valid_address(y_address, y_length)
        or not valid_address(z_address, z_length)
        or not valid_address(ax_address, ax_length)
        or not valid_address(px_address, px_length)
        or not valid_address(aty_address, aty_length)
        or not valid_address(rhs_address, rhs_length)
        or not valid_address(info_address, info_length)
        or max_iter < 0
        or check_termination < 0
        or adaptive_interval < 0
        or sigma <= 0.0
    ):
        return -1
    var p = ptr(p_address)
    var a = ptr(a_address)
    var q = ptr(q_address)
    var lower = ptr(lower_address)
    var upper = ptr(upper_address)
    var rho = ptr(rho_address)
    var factor = ptr(factor_address)
    var x = ptr(x_address)
    var y = ptr(y_address)
    var z = ptr(z_address)
    var ax = ptr(ax_address)
    var px = ptr(px_address)
    var aty = ptr(aty_address)
    var rhs = ptr(rhs_address)
    var info = ptr(info_address)

    var primal = 0.0
    var dual = 0.0
    var primal_scale = 0.0
    var dual_scale = 0.0
    var iterations = 0
    var rho_updates = 0
    var status = 7

    for iteration in range(max_iter):
        if diagonal:
            var i = 0
            var sigma_vec = SIMD[DType.float64, W](sigma)
            while i + W <= n:
                rhs.store(
                    i,
                    (
                        sigma_vec * x.load[width=W](i)
                        - q.load[width=W](i)
                        + a.load[width=W](i)
                        * (
                            rho.load[width=W](i) * z.load[width=W](i)
                            - y.load[width=W](i)
                        )
                    )
                    / factor.load[width=W](i),
                )
                i += W
            while i < n:
                rhs[i] = (
                    sigma * x[i]
                    - q[i]
                    + a[i] * (rho[i] * z[i] - y[i])
                ) / factor[i]
                i += 1
        else:
            build_rhs(a, q, rho, x, y, z, rhs, n, m, sigma)
            solve_factor(factor, rhs, n)
        var col = 0
        while col + W <= n:
            x.store(col, rhs.load[width=W](col))
            col += W
        while col < n:
            x[col] = rhs[col]
            col += 1

        if diagonal:
            diagonal_matvec(a, x, ax, n)
        else:
            matvec(a, x, ax, m, n)
        var row = 0
        var alpha_vec = SIMD[DType.float64, W](alpha)
        var one_minus_alpha = SIMD[DType.float64, W](1.0 - alpha)
        while row + W <= m:
            var old_z = z.load[width=W](row)
            var relaxed = alpha_vec * ax.load[width=W](row) + one_minus_alpha * old_z
            var rho_vec = rho.load[width=W](row)
            var projected = relaxed + y.load[width=W](row) / rho_vec
            projected = max(
                lower.load[width=W](row),
                min(upper.load[width=W](row), projected),
            )
            z.store(row, projected)
            y.store(
                row,
                y.load[width=W](row) + rho_vec * (relaxed - projected),
            )
            row += W
        while row < m:
            var relaxed = alpha * ax[row] + (1.0 - alpha) * z[row]
            var projected = relaxed + y[row] / rho[row]
            projected = max(lower[row], min(upper[row], projected))
            z[row] = projected
            y[row] += rho[row] * (relaxed - projected)
            row += 1

        iterations = iteration + 1
        var should_check = iterations == max_iter
        if check_termination > 0 and iterations % check_termination == 0:
            should_check = True
        if should_check:
            (
                primal,
                dual,
                primal_scale,
                dual_scale,
            ) = compute_residuals(
                p, a, q, z, x, y, ax, px, aty, n, m, diagonal
            )
            if primal <= eps_abs + eps_rel * primal_scale and dual <= eps_abs + eps_rel * dual_scale:
                status = 1
                break

        if adaptive_interval > 0 and iterations % adaptive_interval == 0 and rho_updates < 10:
            if primal == 0.0 and dual == 0.0:
                (
                    primal,
                    dual,
                    primal_scale,
                    dual_scale,
                ) = compute_residuals(
                    p, a, q, z, x, y, ax, px, aty, n, m, diagonal
                )
            var scale = 1.0
            if primal > adaptive_tolerance * dual and dual > 0.0:
                scale = min(5.0, sqrt(primal / dual))
            elif dual > adaptive_tolerance * primal and primal > 0.0:
                scale = max(0.2, sqrt(primal / dual))
            if scale != 1.0:
                for row in range(m):
                    rho[row] = min(1.0e6, max(1.0e-6, rho[row] * scale))
                if not factor_system(
                    p, a, rho, factor, n, m, sigma, diagonal
                ):
                    status = 9
                    break
                rho_updates += 1
                primal = 0.0
                dual = 0.0

    (
        primal,
        dual,
        primal_scale,
        dual_scale,
    ) = compute_residuals(
        p, a, q, z, x, y, ax, px, aty, n, m, diagonal
    )
    var objective = 0.5 * dot(x, px, n) + dot(q, x, n)
    info[0] = Float64(iterations)
    info[1] = objective
    info[2] = primal
    info[3] = dual
    info[4] = Float64(rho_updates)
    if m > 0:
        info[5] = rho[0]
    else:
        info[5] = 0.0
    return status
