import numpy as np

class EmbeddingData:
#    // data
#    pub n: usize,
#    pub d: usize,
#    pub X: Array2<f64>,
#    pub T: Array1<f64>,
#    pub I: Array1<bool>,
#    pub f0_vals: Option<Array1<f64>>,
#    pub N: Array1<f64>,
#    pub R: Array1<usize>,
#    pub Z: Array1<usize>,
#    pub R_bar: Array1<f64>,
#    pub ln_cent: f64,
#    // method
#    pub method: Method,
#    // kernel
#    pub norm_one: Option<f64>,
#    pub K: Option<Array2<f64>>,
#    pub K_bar: Option<Array1<f64>>,
#    pub K_cent: Option<Array2<f64>>,
#    pub K_hat: Option<Array2<f64>>,
#    // feature map
#    pub feature_dim: Option<usize>,
#    pub feature_const: Option<f64>,
#    pub Phi: Option<Array2<f64>>,
#    pub Phi_bar: Option<Array1<f64>>,
#    pub Phi_cent: Option<Array2<f64>>,

    def __init__(self, data, kernel, method):
        data.sort();
        #print(data.X)
        #print(data.T)
#        let n = data_sorted.n;
#        let n64 = n as f64;
#        let d = data_sorted.d;
#        let X = data_sorted.X;
#        let T = data_sorted.T;
#        let I = data_sorted.I;
#        let N = Array1::from_shape_fn(n, |i| f64::from(!I[i]));
#
        data.T[2] = data.T[3]

        # R
        # TODO check this agrees with Rust
        self.R = np.zeros(data.n)
        R_prev = 0
        for j in range(data.n):
            candidates = np.arange(R_prev, j + 1)
            mask = data.T[candidates] >= data.T[j]
            R_val = candidates[np.argmax(mask)]
            self.R[j] = R_val
            R_prev = R_val

        # Z
        # TODO check this agrees with Rust
        self.Z = np.zeros(data.n)
        Z_prev = data.n
        for i in reversed(range(data.n)):
            candidates = np.arange(i, Z_prev)
            mask = data.T[i] >= data.T[candidates]
            Z_val = candidates[mask][-1]
            self.Z[i] = Z_val
            Z_prev = Z_val

        #print(self.R)
        #print(self.Z)




#
#        let R_bar = Array1::from_shape_fn(n, |i| ((n - R[i]) as f64) / n64);
#        let ln_cent = (1..n).map(|i| R_bar[i].ln() * N[i]).sum::<f64>() / n64;
#
#        // kernel
#        let norm_one = match method {
#            Method::Kernel => Some(kernel.norm_one()),
#            Method::FeatureMap => None,
#        };
#        let K = match method {
#            Method::Kernel => Some(Array2::from_shape_fn((n, n), |(i, j)| {
#                kernel.k(X.row(i), X.row(j))
#            })),
#            Method::FeatureMap => None,
#        };
#        let K_bar = match method {
#            Method::Kernel => Some(K.as_ref().unwrap().sum_axis(Axis(0)) / n64),
#            Method::FeatureMap => None,
#        };
#        let K_cent = match method {
#            Method::Kernel => Some(Array2::from_shape_fn((n, n), |(i, j)| {
#                K.as_ref().unwrap()[[i, j]] - K_bar.as_ref().unwrap()[j]
#            })),
#            Method::FeatureMap => None,
#        };
#        let K_hat = match method {
#            Method::Kernel => Some(Array2::from_shape_fn((n, n), |(i, j)| {
#                K.as_ref().unwrap()[[i, j]]
#                    - K_bar.as_ref().unwrap()[i]
#                    - K_bar.as_ref().unwrap()[j]
#                    + K_bar.as_ref().unwrap()[i]
#                        * K_bar.as_ref().unwrap()[j]
#                        * kernel.norm_one()
#            })),
#            Method::FeatureMap => None,
#        };
#
#        // feature map
#        let feature_dim = match method {
#            Method::Kernel => None,
#            Method::FeatureMap => Some(kernel.feature_dim(d)),
#        };
#        let feature_const = match method {
#            Method::Kernel => None,
#            Method::FeatureMap => Some(kernel.feature_const()),
#        };
#        let Phi = match method {
#            Method::Kernel => None,
#            Method::FeatureMap => {
#                let feature_dim = kernel.feature_dim(d);
#                Some(Array2::from_shape_fn((n, feature_dim), |(i, j)| {
#                    let phi_i = kernel.phi(X.row(i));
#                    phi_i[j]
#                }))
#            }
#        };
#        let Phi_bar = match method {
#            Method::Kernel => None,
#            Method::FeatureMap => {
#                Some(Phi.as_ref().unwrap().sum_axis(Axis(0)) / n64)
#            }
#        };
#        let Phi_cent = match method {
#            Method::Kernel => None,
#            Method::FeatureMap => {
#                let feature_dim = kernel.feature_dim(d);
#                Some(Array2::from_shape_fn((n, feature_dim), |(i, j)| {
#                    Phi.as_ref().unwrap()[[i, j]] - Phi_bar.as_ref().unwrap()[j]
#                }))
#            }
#        };
#
#        EmbeddingData {
#            n,
#            d,
#            X,
#            T,
#            I,
#            f0_vals: data_sorted.f0_vals,
#            N,
#            R,
#            Z,
#            R_bar,
#            ln_cent,
#            method,
#            norm_one,
#            K,
#            K_bar,
#            K_cent,
#            K_hat,
#            feature_dim,
#            feature_const,
#            Phi,
#            Phi_bar,
#            Phi_cent,
#        }
#    }





#}
#
##[derive(Clone, Debug)]
class Embedding:
    pass
#    pub train: EmbeddingData,
#    pub valid: EmbeddingData,
#    pub test: EmbeddingData,
#    pub K_cent_valid_train: Option<Array2<f64>>,
#    pub K_cent_test_train: Option<Array2<f64>>,
#}
#
#impl EmbeddingData {
#    #[allow(clippy::too_many_lines)]
#}
#
#impl Embedding {
#    pub fn new(
#        data_train: &Data,
#        data_valid: &Data,
#        data_test: &Data,
#        kernel: &impl Kernel,
#        method: Method,
#    ) -> Self {
#        let train = EmbeddingData::new(data_train, kernel, method);
#        let valid = EmbeddingData::new(data_valid, kernel, method);
#        let test = EmbeddingData::new(data_test, kernel, method);
#        let K_cent_valid_train = match method {
#            Method::Kernel => {
#                Some(Array2::from_shape_fn((valid.n, train.n), |(i, j)| {
#                    kernel.k(valid.X.row(i), train.X.row(j))
#                        - train.K_bar.as_ref().unwrap()[j]
#                }))
#            }
#            Method::FeatureMap => None,
#        };
#        let K_cent_test_train = match method {
#            Method::Kernel => {
#                Some(Array2::from_shape_fn((test.n, train.n), |(i, j)| {
#                    kernel.k(test.X.row(i), train.X.row(j))
#                        - train.K_bar.as_ref().unwrap()[j]
#                }))
#            }
#            Method::FeatureMap => None,
#        };
#
#        Embedding {
#            train,
#            valid,
#            test,
#            K_cent_valid_train,
#            K_cent_test_train,
#        }
#    }
#}
