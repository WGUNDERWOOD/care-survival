#use ndarray::{Array1, ArrayView1};
#
#pub trait Kernel {
#    fn k(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64;
#    fn norm_one(&self) -> f64;
#    fn phi(&self, x: ArrayView1<f64>) -> Array1<f64>;
#    fn feature_dim(&self, d: usize) -> usize;
#    fn feature_const(&self) -> f64;
#}
#
#// Shifted Gaussian kernel
#
#pub struct ShiftedGaussianKernel {
#    pub a: f64,
#    pub sigma: f64,
#}
#
#impl ShiftedGaussianKernel {
#    #[must_use]
#    pub fn new(a: f64, sigma: f64) -> Self {
#        Self { a, sigma }
#    }
#}
#
#impl Kernel for ShiftedGaussianKernel {
#    fn k(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
#        let d = x.len();
#        self.a
#            + (-(0..d)
#                .map(|i| ((x[i] - y[i]) / self.sigma).powf(2.0))
#                .sum::<f64>()
#                / 1.0)
#                .exp()
#    }
#
#    fn norm_one(&self) -> f64 {
#        (1.0 / self.a).powf(0.5)
#    }
#
#    fn phi(&self, _x: ArrayView1<f64>) -> Array1<f64> {
#        panic!()
#    }
#
#    fn feature_dim(&self, _d: usize) -> usize {
#        panic!()
#    }
#
#    fn feature_const(&self) -> f64 {
#        panic!()
#    }
#}
#
#// Polynomial Kernel
#
#pub struct PolynomialKernel {
#    pub a: f64,
#    pub p: usize,
#}
#
#impl PolynomialKernel {
#    #[must_use]
#    pub fn new(a: f64, p: usize) -> Self {
#        Self { a, p }
#    }
#}
#
#impl Kernel for PolynomialKernel {
#    fn k(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
#        let d64 = x.len() as f64;
#        (x.dot(&y) / d64 + self.a).powf(self.p as f64)
#    }
#
#    fn norm_one(&self) -> f64 {
#        (1.0 / self.a).powf(self.p as f64 / 2.0)
#    }
#
#    fn phi(&self, x: ArrayView1<f64>) -> Array1<f64> {
#        let d = x.len();
#        let feature_dim = self.feature_dim(d);
#        match self.p {
#            1 => x.to_owned(),
#            2 => {
#                let mut phi_x = vec![f64::NAN; feature_dim];
#                let sqrt_2 = (2.0_f64).sqrt();
#                let d64 = d as f64;
#                let sqrt_d = d64.sqrt();
#                let mut r = 0;
#                for i in 0..d {
#                    phi_x[r] = sqrt_2 * x[i] / sqrt_d;
#                    r += 1;
#                }
#                for i in 0..d {
#                    phi_x[r] = x[i] * x[i] / d64;
#                    r += 1;
#                }
#                for i in 0..d {
#                    for j in 0..i {
#                        phi_x[r] = sqrt_2 * x[i] * x[j] / d64;
#                        r += 1;
#                    }
#                }
#                phi_x.into()
#            }
#            _ => panic!(),
#        }
#    }
#
#    fn feature_dim(&self, d: usize) -> usize {
#        match self.p {
#            1 => d,
#            2 => 2 * d + d * (d - 1) / 2,
#            _ => panic!(),
#        }
#    }
#
#    fn feature_const(&self) -> f64 {
#        self.a.powf(self.p as f64 / 2.0)
#    }
#}
#
#// Shifted first order Sobolev kernel
#
#pub struct ShiftedFirstOrderSobolevKernel {
#    pub a: f64,
#}
#
#impl ShiftedFirstOrderSobolevKernel {
#    #[must_use]
#    pub fn new(a: f64) -> Self {
#        Self { a }
#    }
#}
#
#impl Kernel for ShiftedFirstOrderSobolevKernel {
#    fn k(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
#        let d = x.len();
#        self.a + (0..d).map(|i| x[i].min(y[i])).sum::<f64>()
#    }
#
#    fn norm_one(&self) -> f64 {
#        (1.0 / self.a).powf(0.5)
#    }
#
#    fn phi(&self, _x: ArrayView1<f64>) -> Array1<f64> {
#        panic!()
#    }
#
#    fn feature_dim(&self, _d: usize) -> usize {
#        panic!()
#    }
#
#    fn feature_const(&self) -> f64 {
#        panic!()
#    }
#}
