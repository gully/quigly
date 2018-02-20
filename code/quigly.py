import numpy as np

def get_C_j(phi_a, phi_b, phi_c):
    '''Return the GMM covariance matrix given phi parameters'''
    v11 = np.exp(2.0*phi_a)
    v12 = phi_c*np.exp(phi_a)
    v22 = phi_c**2 + np.exp(2.0*phi_b)
    C_j = np.matrix([[v11, v12],[v12, v22]])
    return C_j

def get_gmm_j(x, y, x_j, y_j, C_j):
    '''Returns the j^th normalized Gaussian mixture model'''
    x = x - x_j
    y = y - y_j
    rr = np.stack([x,y], axis=0)
    exp_arg = np.sum(rr * np.linalg.solve(C_j, rr), axis=0)
    gmm_j_raw = np.exp(-0.5 * exp_arg)
    normalization = 2.0*np.pi*np.sqrt(np.linalg.det(C_j))
    gmm_j = 1.0/normalization * gmm_j_raw
    return gmm_j


def gmm_z_js(theta_js):
    '''Return the GMM z_j's given components and theta_js'''
    z_js = np.ones(len(theta_js)+1)
    for j, theta_j in enumerate(theta_js):
        z_js[j+1] = z_js[j] + np.exp(-theta_js[j])
    z_js = z_js / np.sum(z_js)
    return z_js


def compute_model_a(x, y, z_js, x_js, y_js, phi_as, phi_bs, phi_cs):
    '''Takes in x, y and GMM kernel params, returns model'''
    N_GMM, = x_js.shape
    gmm_out = 0.0
    for j in range(N_GMM):
        C_j = get_C_j(phi_as[j], phi_bs[j], phi_cs[j])
        gmm_out += z_js[j] * get_gmm_j(x, y, x_js[j], y_js[j], C_j)
    return gmm_out

def get_f_star_nocov(A, sigma_r, d):
    '''Compute the profile likelihood of the stellar fluxes'''
    #For now ignore the covariance matrix, assume homoscedastic
    ATA = np.dot(A.T, A / sigma_r)
    f_star = np.linalg.solve(ATA, np.dot(A.T, d/sigma_r))
    return f_star

def get_A_matrix(x, y, x_c, y_c, kernel_params):
    '''Return the column-wise concatenated A matrix'''
    N_stars = len(x_c)
    xx = x[:, np.newaxis] - x_c.T
    yy = y[:, np.newaxis] - y_c.T
    A_matrix = np.zeros(xx.shape)
    for i in range(N_stars):
        z_js, x_js, y_js, phi_as, phi_bs, phi_cs = kernel_params.T
        A_matrix[:, i] = compute_model_a(xx[:,i], yy[:,i],
                                         z_js, x_js, y_js, phi_as, phi_bs, phi_cs)
    return A_matrix

def split_params(params, N_star, N_GMM):
    '''Split/clean all parameters into star and Kernel parameters'''
    star_params = params[0:N_star*2].reshape((N_star, -1))
    fixed_gmm_params = np.array([1, 0, 0])
    kern_params = np.hstack([fixed_gmm_params, params[N_star*2:]]).reshape((N_GMM, -1))
    kern_params[:, 0] = gmm_z_js(kern_params[1:, 0])
    return star_params, kern_params

def lnlike(params):
    '''Return the likelihood given the parameters'''
    star_params, kern_params = split_params(params, 4, 3)

    x_c, y_c = star_params[0, :], star_params[1, :]

    # Just a homoscedastic noise matrix for now
    #C_noise_matrix = get_C_matrix(sigma_r, N_pix)

    # Get the design matrix for stellar fluxes
    A_matrix = get_A_matrix(x, y, x_c, y_c, kern_params)

    # Compute the profile likelihood for stellar fluxes
    f_star = get_f_star_nocov(A_matrix, sigma_r, data)

    model = np.dot(A_matrix, f_star)
    resid = data - model
    lnlike_out = np.dot(resid.T, resid / yerr**2)
    return lnlike_out