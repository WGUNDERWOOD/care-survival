##![allow(non_snake_case)]
#
#use dirs::home_dir;
#use polars::prelude::*;
#use std::fs::File;
#
#fn main() {
#    let mut dir = home_dir().unwrap();
#    dir = dir.join("rds/rds-ceu-ukbiobank-RtePkTecWB4");
#    dir = dir.join("projects/P7439/lambertlab/wgu21/data");
#    let path = dir.join("df_SCORE2_withexclusions.csv");
#    let mut df = LazyCsvReader::new(path).finish().unwrap();
#    df = df.filter(col("imd_country").eq(lit("England")));
#    df = df
#        .lazy()
#        // add and rename transformed features
#        .with_columns([
#            ((col("ages") - lit(60.0)) / lit(5.0)).alias("cage"),
#            ((col("hdl") - lit(1.3)) / lit(0.5)).alias("chdl"),
#            ((col("sbp") - lit(120.0)) / lit(20.0)).alias("csbp"),
#            ((col("tchol") - lit(6.0)) / lit(1.0)).alias("ctchol"),
#            when(col("smallbin").eq(lit("Current")))
#                .then(lit(1.0))
#                .otherwise(lit(0.0))
#                .alias("smoking"),
#            col("eGFRcreat").alias("egfrcreat"),
#            col("imd_min").alias("imd"),
#            col("PGS000018").alias("pgs000018"),
#            col("PGS000039").alias("pgs000039"),
#            col("FOLLOWUPTIME_Incident_10year").alias("time"),
#            col("PHENOTYPE_Incident_10year").alias("indicator"),
#        ])
#        // add interaction terms
#        .with_columns([
#            (col("cage") * col("chdl")).alias("cage_chdl"),
#            (col("cage") * col("csbp")).alias("cage_csbp"),
#            (col("cage") * col("ctchol")).alias("cage_ctchol"),
#            (col("cage") * col("smoking")).alias("cage_smoking"),
#        ])
#        // compute SCORE2 predictions
#        .with_column(
#            when(col("sex").eq(lit("Male")))
#                .then(
#                    col("cage") * lit(1.50_f64.ln())
#                        + col("smoking") * lit(1.77_f64.ln())
#                        + col("csbp") * lit(1.33_f64.ln())
#                        + col("ctchol") * lit(1.13_f64.ln())
#                        + col("chdl") * lit(0.80_f64.ln())
#                        + col("cage_smoking") * lit(0.92_f64.ln())
#                        + col("cage_csbp") * lit(0.98_f64.ln())
#                        + col("cage_ctchol") * lit(0.98_f64.ln())
#                        + col("cage_chdl") * lit(1.04_f64.ln()),
#                )
#                .otherwise(
#                    col("cage") * lit(1.64_f64.ln())
#                        + col("smoking") * lit(2.09_f64.ln())
#                        + col("csbp") * lit(1.39_f64.ln())
#                        + col("ctchol") * lit(1.11_f64.ln())
#                        + col("chdl") * lit(0.81_f64.ln())
#                        + col("cage_smoking") * lit(0.89_f64.ln())
#                        + col("cage_csbp") * lit(0.97_f64.ln())
#                        + col("cage_ctchol") * lit(0.98_f64.ln())
#                        + col("cage_chdl") * lit(1.06_f64.ln()),
#                )
#                .alias("score2_rel"),
#        );
#
#    let rescale_cols = [
#        "cage",
#        "chdl",
#        "csbp",
#        "ctchol",
#        "cage_chdl",
#        "cage_csbp",
#        "cage_ctchol",
#        "cage_smoking",
#        "crp",
#        "egfrcreat",
#        "imd",
#        "pgs000018",
#        "pgs000039",
#    ];
#
#    let df_min = df.clone().min().collect().unwrap();
#    let df_max = df.clone().max().collect().unwrap();
#
#    // rescale cols
#    for c in rescale_cols {
#        let col_min = df_min.column(c).unwrap().f64().unwrap().get(0).unwrap();
#        let col_max = df_max.column(c).unwrap().f64().unwrap().get(0).unwrap();
#        let col_range = col_max - col_min;
#        let col_name = format!("{c}_scaled");
#        df = df.lazy().with_columns([((col(c) - lit(col_min))
#            / lit(col_range))
#        .alias(col_name)]);
#    }
#
#    // select and rename cols
#    df = df.lazy().select([
#        col("sex"),
#        col("cage_scaled").alias("age"),
#        col("chdl_scaled").alias("hdl"),
#        col("csbp_scaled").alias("sbp"),
#        col("ctchol_scaled").alias("tchol"),
#        col("smoking"),
#        col("cage_chdl_scaled").alias("age_hdl"),
#        col("cage_csbp_scaled").alias("age_sbp"),
#        col("cage_ctchol_scaled").alias("age_tchol"),
#        col("cage_smoking_scaled").alias("age_smoking"),
#        col("crp_scaled").alias("crp"),
#        col("egfrcreat_scaled").alias("egfrcreat"),
#        col("imd_scaled").alias("imd"),
#        col("pgs000018_scaled").alias("pgs000018"),
#        col("pgs000039_scaled").alias("pgs000039"),
#        col("score2_rel"),
#        (col("time") / lit(10.0)).alias("time"),
#        (lit(1.0) - col("indicator"))
#            .cast(DataType::Boolean)
#            .alias("censored"),
#    ]);
#
#    let df_male = df
#        .clone()
#        .filter(col("sex").eq(lit("Male")))
#        .drop([col("sex")])
#        .drop_nulls(None);
#    let df_female = df
#        .clone()
#        .filter(col("sex").eq(lit("Female")))
#        .drop([col("sex")])
#        .drop_nulls(None);
#
#    let out_path_male = dir.join("df_scaled_male.csv");
#    let out_path_female = dir.join("df_scaled_female.csv");
#
#    let mut out_file_male = File::create(out_path_male).unwrap();
#    let mut out_file_female = File::create(out_path_female).unwrap();
#
#    CsvWriter::new(&mut out_file_male)
#        .include_header(true)
#        .with_separator(b',')
#        .finish(&mut df_male.collect().unwrap())
#        .unwrap();
#
#    CsvWriter::new(&mut out_file_female)
#        .include_header(true)
#        .with_separator(b',')
#        .finish(&mut df_female.collect().unwrap())
#        .unwrap();
#}
