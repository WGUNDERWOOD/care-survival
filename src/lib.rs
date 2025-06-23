use pyo3::prelude::*;
use crate::example::*;

mod data;
mod example;

#[pymodule]
fn care_survival(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
