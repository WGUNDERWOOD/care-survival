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
