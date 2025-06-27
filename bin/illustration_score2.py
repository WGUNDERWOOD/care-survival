import sys

from care_survival import score2 as care_score2
from care_survival import kernels as care_kernels
from care_survival import embedding as care_embedding

def main():
    sex = sys.argv[1]
    n_female = 162682
    n_male = 121333
    if sex == "female":
        n_train = n_female
    elif sex == "male":
        n_train = n_male
    n_valid = 0
    n_test = 0
    covs = ["imd"]
    dry_run = True
    method = "feature_map"

    rep = 0
    data_train, data_valid, data_test, score2_rel = care_score2.get_score2_data(n_train, n_valid, n_test, covs, sex, dry_run, rep)

    # kernel and embedding
    a = 1
    p = 1
    kernel = care_kernels.PolynomialKernel(a, p)
    embedding = care_embedding.Embedding(
        data_train, data_valid, data_test, kernel, method
    )

    # TODO write this file

    exit()

    #// get high and low imd data sets
    #let mut imds: Vec<f64> =
    #    (0..n_train).map(|i| embedding.train.X[[i, 0]]).collect()
    #imds.sort_by(|x, y| x.total_cmp(y))
    #let imd_med = imds[n_train.div_ceil(2)]
    #let high_is: Vec<usize> = (0..n_train)
    #    .filter(|&i| embedding.train.X[[i, 0]] >= imd_med)
    #    .collect()
    #let low_is: Vec<usize> = (0..n_train)
    #    .filter(|&i| embedding.train.X[[i, 0]] < imd_med)
    #    .collect()
    #let data_train_high = Data {
    #    n: high_is.len(),
    #    d: covs.len(),
    #    X: embedding.train.X.select(Axis(0), &high_is),
    #    T: embedding.train.T.select(Axis(0), &high_is),
    #    I: embedding.train.I.select(Axis(0), &high_is),
    #    f0_vals: None,
    #}
    #let data_train_low = Data {
    #    n: low_is.len(),
    #    d: covs.len(),
    #    X: embedding.train.X.select(Axis(0), &low_is),
    #    T: embedding.train.T.select(Axis(0), &low_is),
    #    I: embedding.train.I.select(Axis(0), &low_is),
    #    f0_vals: None,
    #}
    #let embedding_high = Embedding::new(
    #    &data_train_high,
    #    &data_valid,
    #    &data_test,
    #    &kernel,
    #    method,
    #)
    #let embedding_low = Embedding::new(
    #    &data_train_low,
    #    &data_valid,
    #    &data_test,
    #    &kernel,
    #    method,
    #)

    #// get breslow estimator
    #let breslow = embedding.train.get_breslow()
    #let mut breslow_high = embedding_high.train.get_breslow().into_iter()
    #let mut breslow_low = embedding_low.train.get_breslow().into_iter()
    #let mut s: String = "T,I,imd,breslow,breslow_high,breslow_low\n".into()
    #for i in 0..n_train {
    #    if embedding.train.T[i] < 1.0 {
    #        s.push_str(&embedding.train.T[i].to_string())
    #        s.push(',')
    #        s.push_str(&embedding.train.I[i].to_string())
    #        s.push(',')
    #        s.push_str(&embedding.train.X[[i, 0]].to_string())
    #        s.push(',')
    #        s.push_str(&breslow[i].to_string())
    #        s.push(',')
    #        if high_is.contains(&i) {
    #            s.push_str(&breslow_high.next().unwrap().to_string())
    #        } else {
    #            s.push_str("NA")
    #        }
    #        s.push(',')
    #        if low_is.contains(&i) {
    #            s.push_str(&breslow_low.next().unwrap().to_string())
    #        } else {
    #            s.push_str("NA")
    #        }
    #        s.push('\n')
    #    }
    #}

    #// write breslow results
    #let file_name = format!("illustration_score2_{sex}.csv")
    #let path = current_dir()
    #    .unwrap()
    #    .join("data")
    #    .join(format!("{}", Local::now().format("%Y-%m-%d")))
    #    .join("score2")
    #    .join(file_name)
    #create_dir_all(path.parent().unwrap()).unwrap()
    #let mut file = File::create(path).unwrap()
    #file.write_all(s.as_bytes()).unwrap()

if __name__ == "__main__":
    main()
