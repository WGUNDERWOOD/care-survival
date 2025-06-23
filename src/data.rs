//use ndarray::{Array1, Array2};

#[derive(Debug)]
pub struct Data {
    pub n: usize,
    pub d: usize,
    //pub X: Array2<f64>,
    //pub T: Array1<f64>,
    //pub I: Array1<bool>,
    //pub f0_vals: Option<Array1<f64>>,
}

/*
impl Data {
    #[must_use]
    pub fn sort(&self) -> Self {
        let mut indices: Vec<usize> = (0..self.n).collect();
        indices.sort_by(|&i, &j| self.T[i].total_cmp(&self.T[j]));
        let n = self.n;
        let d = self.d;
        let X = Array2::from_shape_fn((self.n, self.d), |(i, j)| {
            self.X[[indices[i], j]]
        });
        let T = Array1::from_shape_fn(self.n, |i| self.T[indices[i]]);
        let I = Array1::from_shape_fn(self.n, |i| self.I[indices[i]]);
        let f0_vals = self
            .f0_vals
            .as_ref()
            .map(|f| Array1::from_shape_fn(self.n, |i| f[indices[i]]));
        Data {
            n,
            d,
            X,
            T,
            I,
            f0_vals,
        }
    }
}
*/
