use bytemuck;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray_stats::*;

mod ecg_data {
    // use bytemuck;
    use std::fs::File;
    use std::io::Read;
    use std::io::Write;
    use crate::proc_ecg::pre_proc_r;
    struct Pacient {
        time: String,
        date: String,
        pacient: String,
        age: String,
        number_room: String,
        number_history: String,
    }

    pub struct Ecg {
        pub lead1: Vec<f32>,
        pub lead2: Vec<f32>,
        pub lead3: Vec<f32>,
        pub len_lead: usize,
        pub ind_r: Vec<u64>,
        pub intervals_r: Vec<u64>,
        pub div_intervals: Vec<Option<f32>>,
    }

    impl Ecg {
        pub fn read_ecg(&mut self) {
            let files = glob::glob("*.ecg").expect("Failed to read files");
            let fname = files.filter_map(Result::ok).next().unwrap();
            println!("{:?}", fname);

            let mut file = File::open(fname).expect("Failed to open file");
            // let mut buffer = Vec::new();
            let mut buffer = [0; 900000];
            // file.read_to_end(&mut buffer).expect("Failed to read file");
            file.read(&mut buffer[..]).expect("Failed to read file");

            let ecg: Vec<i16> = buffer[1024..]
                .chunks(2)
                .map(|chunk| ((chunk[1] as i32) << 8 | (chunk[0] as i32)) as i16)
                .collect();

            let ch_1: Vec<f32> = ecg
                .iter()
                .step_by(3)
                .map(|&val| (-val as f32 + 1024.0) / 100.0)
                .collect();

            let ch_2: Vec<f32> = ecg
                .iter()
                .skip(1)
                .step_by(3)
                .map(|&val| (-val as f32 + 1024.0) / 100.0)
                .collect();

            let ch_3: Vec<f32> = ecg
                .iter()
                .skip(2)
                .step_by(3)
                .map(|&val| (-val as f32 + 1024.0) / 100.0)
                .collect();

            let mean_ch_1: f32 = ch_1.iter().sum::<f32>() / ch_1.len() as f32;
            let mean_ch_2: f32 = ch_2.iter().sum::<f32>() / ch_2.len() as f32;
            let mean_ch_3: f32 = ch_3.iter().sum::<f32>() / ch_3.len() as f32;

            let ch_1: Vec<f32> = ch_1.iter().map(|&val| val - mean_ch_1).collect();
            let ch_2: Vec<f32> = ch_2.iter().map(|&val| val - mean_ch_2).collect();
            let ch_3: Vec<f32> = ch_3.iter().map(|&val| val - mean_ch_3).collect();

            self.lead1 = ch_1;
            self.lead2 = ch_2;
            self.lead3 = ch_3;
            self.len_lead = self.lead1.len();
            let sum_leads = pre_proc_r(&self.lead1, &self.lead2, &self.lead3);
            self.get_ind_r(&sum_leads);
            self.get_div_intervals();
        }
        fn get_ind_r(&mut self, ch: &Vec<f32>) {
            let mut max_val: f32 = 0.0;
            let mut ind_max = 0;
            let mut interval: u64 = 0;
            for (ind, val) in ch.iter().enumerate() {
                if *val > max_val {
                    max_val = *val;
                    ind_max = ind as u64;
                }
                if *val <= 0.0 && max_val > 0.0 {
                    max_val = 0.0;
                    // self.ind_r.push(ind_max);
                    if self.intervals_r.len() == 0 {
                        self.intervals_r.push(ind_max);
                    } else {
                        interval = ind_max - self.ind_r[self.ind_r.len() - 1];
                        self.intervals_r.push(interval);
                    }
                    self.ind_r.push(ind_max);
                }
            }
        }
        fn get_div_intervals(&mut self) {
            if self.intervals_r.len() > 0 {
                let mut intervals = self.intervals_r.to_owned();
                intervals.push(self.intervals_r[self.intervals_r.len() - 1]);
                intervals.remove(0);
                self.div_intervals = self.intervals_r
                    .iter()
                    .zip(intervals.iter())
                    .map(|(&x, &y)| {
                        if y > 0 {
                            Some(x as f32 / y as f32)
                        } else {
                            None
                        }
                    })
                    .collect();
            }
        }
    }
}

mod proc_ecg {
    fn cut_impuls(ch: &Vec<f32>) -> Vec<f32> {
        let mut out = ch.clone();
        let mut win = [0.0; 11];
        let lench = ch.len();

        for i in (3..lench - 5).step_by(1) {
            let j = i % 11;
            win[j] = ch[i];
            let d1 = ch[i - 1] - ch[i - 2];
            let d2 = ch[i] - ch[i - 1];
            let d3 = ch[i + 1] - ch[i];

            if (ch[i] - out[i - 1]).abs() > 1.95 {
                let mut sort_win = win.to_vec();
                sort_win.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[i - 2] = sort_win[5];
                out[i - 1] = sort_win[5];
                out[i] = sort_win[5];
                out[i + 1] = sort_win[5];
                out[i + 2] = sort_win[5];
                out[i + 3] = sort_win[5];
                out[i + 4] = sort_win[5];
                out[i + 5] = sort_win[5];
            }
            if (d1.signum() != d2.signum()) && (d2.signum() != d3.signum()) && ((d1.abs() + d2.abs() + d3.abs()) > 0.7) {
                let mut sort_win = win.to_vec();
                sort_win.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[i - 2] = sort_win[5];
                out[i - 1] = sort_win[5];
                out[i] = sort_win[5];
                out[i + 1] = sort_win[5];
                out[i + 2] = sort_win[5];
            }
        }
        out
    }
    fn my_filtfilt(b: &Vec<f32>, a: &Vec<f32>, ch: &Vec<f32>) -> Vec<f32> {
        let mut temp = ch.to_owned();
        let mut out = ch.to_owned();
        let len_b = b.len();
        let len_a = a.len();
        let len_ch = ch.len();

        for i in (len_b - 1..len_ch).step_by(1) {
            temp[i] = b[0] * ch[i];
            for j in 1..len_b {
                temp[i] += b[j] * ch[i - j];
            }
            for j in 1..len_a {
                temp[i] -= a[j] * temp[i - j];
            }
        }

        for i in (1..=(len_ch - len_b)).rev() {
            out[i] = b[0] * temp[i];
            for j in 1..len_b {
                out[i] += b[j] * temp[i + j];
            }
            for j in 1..len_a {
                out[i] -= a[j] * out[i + j];
            }
        }
        out
    }
    fn get_spec24(ch: &Vec<f32>) -> Vec<f32> {
        let bp = vec![0.05695238, 0.0, -0.05695238];
        let ap = vec![1.0, -1.55326091, 0.88609524];

        let bl = vec![0.11216024, 0.11216024];
        let al = vec![1.0, -0.77567951];

        let bh = vec![0.97547839, -0.97547839];
        let ah = vec![1.0, -0.95095678];

        let spec24 = my_filtfilt(&bp, &ap, &ch);
        let spec24 = spec24.iter().map(|&x| (x * 4.0).abs()).collect::<Vec<f32>>();
        let spec24 = my_filtfilt(&bl, &al, &spec24);
        let spec24 = my_filtfilt(&bh, &ah, &spec24);

        spec24
    }
    fn get_spec50(ch: &Vec<f32>) -> Vec<f32> {
        let bp = vec![0.13672874, 0.0, -0.13672874];
        let ap = vec![1.0, -0.53353098, 0.72654253];

        let bl = vec![0.24523728, 0.24523728];
        let al = vec![1.0, -0.50952545];

        let spec50 = my_filtfilt(&bp, &ap, &ch);
        let spec50 = spec50.iter().map(|&x| x.abs()).collect::<Vec<f32>>();
        let spec50 = my_filtfilt(&bl, &al, &spec50);

        spec50
    }
    fn clean_ch(ch: &Vec<f32>) -> Vec<f32> {
        let b = vec![0.0461318, 0.0922636, 0.0461318];
        let a = vec![1.0, -1.30728503, 0.49181224];
        // let mut proc = Proc {};
        let ch_del_ks = cut_impuls(&ch);
        let mut fch = ch_del_ks.clone();

        let spec24 = get_spec24(&ch_del_ks);
        let spec50 = get_spec50(&ch_del_ks);
        let clean_ch = my_filtfilt(&b, &a, &ch_del_ks);

        for i in 0..fch.len() {
            if spec50[i] > spec24[i] {
                fch[i] = clean_ch[i];
            }
        }
        fch
    }
    fn get_p2p(ch: &Vec<f32>, win: usize, sqr: bool) -> Vec<f32> {
        let half_win = win / 2;
        let len_ch = ch.len();
        let mut p2p = vec![0.0; len_ch];
        for i in (0..len_ch - &win).step_by(5) {
            let win_ch = &ch[i..i + &win];
            let win_max = win_ch.iter().fold(f32::NEG_INFINITY, |max, &x| x.max(max));
            let win_min = win_ch.iter().fold(f32::INFINITY, |min, &x| x.min(min));
            let mut p2pw = &win_max - &win_min;
            if sqr {
                p2pw = &p2pw * &p2pw * 1.5;
                if p2pw > 2.0 { p2pw = 2.0; }
            }
            p2p[i + half_win - 2] = p2pw;
            p2p[i + half_win - 1] = p2pw;
            p2p[i + half_win] = p2pw;
            p2p[i + half_win + 1] = p2pw;
            p2p[i + half_win + 2] = p2pw;
        }
        p2p
    }
    fn sign(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
    fn vec_sign(ch: &Vec<f32>) -> Vec<f32> {
        let out = ch.iter().map(|&val| sign(val)).collect();
        out
    }
    fn diff(ch: &Vec<f32>) -> Vec<f32> {
        let ch_copy = ch.to_owned();
        let mut ch_copy1 = ch.to_owned();
        ch_copy1.remove(0);
        ch_copy1.push(0.0);
        let diff_ch: Vec<f32> = ch_copy.iter().zip(ch_copy1.iter()).map(|(val1, val2)| val1 - val2).collect();
        diff_ch
    }
    fn get_trs(p2p: &Vec<f32>) -> f32 {
        let sum_p2p: f32 = p2p
            .iter()
            .sum();
        let mean_p2p: f32 = sum_p2p / p2p.len() as f32;
        let (count, sum): (usize, f32) = p2p
            .iter()
            .filter(|&x| *x > mean_p2p)
            .fold((0, 0.0), |(count, sum), &x| (count + 1, sum + x));
        let mut trs = sum / count as f32;
        if trs < 0.7 {
            trs = 0.7;
        }
        trs
    }
    fn del_artifacts(ch: &Vec<f32>, p2p: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let len_ch = ch.len();
        let mut out = ch.to_owned();
        let mut mask = vec![1.0; len_ch];

        let signs = vec_sign(&ch);
        let diff_signs: Vec<f32> = diff(&signs);

        let mut prev_ind = 0;
        let mut flag: bool = false;
        let trs = get_trs(&p2p);

        for i in 0..diff_signs.len() {
            if p2p[i] > trs * 3.8 {                 // 5.5
                flag = true;
            }
            if diff_signs[i] != 0.0 {
                if flag == true {
                    let zeros = vec![0.0; i - prev_ind + 1];
                    let range = prev_ind..=i;
                    out.splice(range.clone(), zeros.clone());
                    mask.splice(range.clone(), zeros.clone());
                }
                prev_ind = i;
                flag = false;
            }
        }
        (out, mask)
    }
    fn del_nouse(ch: &Vec<f32>, mask: &Vec<f32>) -> Vec<f32> {
        let bhn = vec![0.89884553, -1.79769105, 0.89884553];
        let ahn = vec![1.0, -1.78743252, 0.80794959];
        let bln = vec![0.00034054, 0.00204323, 0.00510806, 0.00681075, 0.00510806, 0.00204323, 0.00034054];
        let aln = vec![1.0, -3.5794348, 5.65866717, -4.96541523, 2.52949491, -0.70527411, 0.08375648];

        let fch = my_filtfilt(&bhn, &ahn, &ch);
        let fch = my_filtfilt(&bln, &aln, &fch);

        let result: Vec<f32> = fch
            .iter()
            .zip(mask.iter())
            .map(|(&x, &y)| x * y)
            .collect();
        result
    }
    fn filt_r(ch: &Vec<f32>) -> Vec<f32> {
        let blr = vec![0.00188141, 0.00188141];
        let alr = vec![1.0, -0.99623718];
        let bhr = vec![0.92867294, -0.92867294];
        let ahr = vec![1.0, -0.85734589];
        let out = my_filtfilt(&blr, &alr, &ch);
        let out = out
            .iter()
            .map(|&x| x * 1000.0)
            .collect();
        let out = my_filtfilt(&bhr, &ahr, &out);
        out
    }
    fn sum_ch(ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>) -> Vec<f32> {
        let result: Vec<f32> = ch1
            .iter()
            .zip(ch2.iter())
            .zip(ch3.iter())
            .map(|((&a, &b), &c)| {
                let sum = a + b + c;
                if sum > 1.5 {
                    1.5
                } else {
                    sum
                }
            })
            .collect();
        result
    }
    pub fn pre_proc_r(ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>) -> Vec<f32> {
        let cln_ch1 = clean_ch(&ch1);
        let cln_ch2 = clean_ch(&ch2);
        let cln_ch3 = clean_ch(&ch3);

        let p2p_ch1 = get_p2p(&cln_ch1, 40, false);
        let p2p_ch2 = get_p2p(&cln_ch2, 40, false);
        let p2p_ch3 = get_p2p(&cln_ch3, 40, false);

        let art1 = del_artifacts(&cln_ch1, &p2p_ch1);
        let art2 = del_artifacts(&cln_ch2, &p2p_ch2);
        let art3 = del_artifacts(&cln_ch3, &p2p_ch3);

        let fch1 = del_nouse(&art1.0, &art1.1);
        let fch2 = del_nouse(&art2.0, &art2.1);
        let fch3 = del_nouse(&art3.0, &art3.1);

        let fch1 = get_p2p(&fch1, 30, true);
        let fch2 = get_p2p(&fch2, 30, true);
        let fch3 = get_p2p(&fch3, 30, true);

        let fch1 = filt_r(&fch1);
        let fch2 = filt_r(&fch2);
        let fch3 = filt_r(&fch3);
        let sum_leads = sum_ch(&fch1, &fch2, &fch3);
        sum_leads
    }
    fn del_isoline(ch: &Vec<f32>) -> Vec<f32> {
        // bi, ai = butter(2, 0.6, 'hp', fs=250)
        let bi = vec![0.98939373, -1.97878745, 0.98939373];
        let ai = vec![1.0, -1.97867496, 0.97889995];
        let out = my_filtfilt(&bi, &ai, &ch);
        out
    }
}

mod qrs_forms {
    use ndarray::Array1;

    pub struct FormsQrs {
        pub ref_form1: Vec<f32>,
        pub ref_form2: Vec<f32>,
        pub ref_form3: Vec<f32>,
    }
    impl FormsQrs {
        fn get_ref_forms(&mut self, ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>, ind_r: &Vec<u64>) {
            for i in 1..ind_r.len() - 1 {
                let start_index = (ind_r[i] - 35) as usize;
                let end_index = (ind_r[i] + 36) as usize;
                let qrs1 = &ch1[start_index..end_index].to_vec();
                let qrs2 = &ch2[start_index..end_index].to_vec();
                let qrs3 = &ch3[start_index..end_index].to_vec();
                let start_index = (ind_r[i + 1] - 35) as usize;
                let end_index = (ind_r[i + 1] + 36) as usize;
                let qrs11 = &ch1[start_index..end_index].to_vec();
                let qrs22 = &ch2[start_index..end_index].to_vec();
                let qrs33 = &ch3[start_index..end_index].to_vec();

                let cor1 = get_coef_cor(qrs1, qrs11);
                let cor2 = get_coef_cor(qrs2, qrs22);
                let cor3 = get_coef_cor(qrs3, qrs33);

                if cor1 > 0.93 && cor2 > 0.93 && cor3 > 0.93 {
                    for i in 0..71 {
                        self.ref_form1[i] = qrs1[i];
                        self.ref_form2[i] = qrs2[i];
                        self.ref_form3[i] = qrs3[i];
                    }
                    break;
                }
            }
        }
    }
    fn norm_qrs(qrs: &Vec<f32>) -> Vec<f32> {
        let mut min:f32 = 0.0;
        let mut max:f32 = 0.0;
        let mut out = qrs.to_owned();
        for i in 0..out.len() {
            if out[i] < min {
                min = out[i];
            }
        }
        for i in 0..out.len() {
            out[i] -= min;
        }
        for i in 0..out.len() {
            if out[i] > max {
                max = out[i];
            }
        }
        if max != 0.0 {
            for i in 0..out.len() {
                out[i] /= max;
            }
        }
        out
    }
    fn get_coef_cor(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        let norm_x = norm_qrs(x);
        let norm_y = norm_qrs(y);
        let arr_x = Array1::from_vec(norm_x.to_vec());
        let arr_y = Array1::from_vec(norm_y.to_vec());
        let mean_x: f32 = arr_x.mean().unwrap();
        let mean_y: f32 = arr_y.mean().unwrap();
        let arr_xy = &arr_x * &arr_y;
        let mean_xy = Array1::from(arr_xy).mean().unwrap();
        let std_x = &arr_x.std(0.0);
        let std_y = &arr_y.std(0.0);
        let std_xy = std_x * std_y;
        let mut out: f32 = 0.0;
        if std_xy != 0.0 {
            out = (&mean_xy - &mean_x * &mean_y) / &std_xy;
        } else { out = 0.0; }
        out
    }

}


// struct FormsQrs {
//     ref_form1: Vec<f32>,
//     ref_form2: Vec<f32>,
//     ref_form3: Vec<f32>,
// }
//
// impl FormsQrs {
//     fn norm_qrs(&mut self, qrs: &Vec<f32>) -> Vec<f32> {
//         let mut min: f32 = 0.0;
//         let mut max: f32 = 0.0;
//         for item in qrs.iter() {
//             if *item < min {
//                 min = *item;
//             }
//         }
//         for i in 0..qrs.len() {
//             &qrs[i] = &qrs[i] - &mut min;
//             // &qrs[i] -= min;
//         }
//         for item in qrs.iter() {
//             if *item > max {
//                 max = *item;
//             }
//         }
//         if max != 0.0 {
//             for i in 0..qrs.len() {
//                 &qrs[i] = &qrs[i] / &mut max;
//                 // qrs[i] /= max;
//             }
//         }
//         qrs.to_vec()
//     }
//     fn get_coef_cor(&mut self, x: &Vec<f32>, y: &Vec<f32>) -> f32 {
//         let norm_x = &self.norm_qrs(x);
//         let norm_y = &self.norm_qrs(y);
//         let arr_x = Array1::from_vec(norm_x.to_vec());
//         let arr_y = Array1::from_vec(norm_y.to_vec());
//         let mean_x: f32 = arr_x.mean().unwrap();
//         let mean_y: f32 = arr_y.mean().unwrap();
//         let arr_xy = &arr_x * &arr_y;
//         let mean_xy = Array1::from(arr_xy).mean().unwrap();
//         let std_x = &arr_x.std(0.0);
//         let std_y = &arr_y.std(0.0);
//         let std_xy = std_x * std_y;
//         let mut out: f32 = 0.0;
//         if std_xy != 0.0 {
//             out = (&mean_xy - &mean_x * &mean_y) / &std_xy;
//         } else { out = 0.0; }
//         out
//     }
//     fn get_ref_forms(&mut self, ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>, ind_r: &Vec<u64>) {
//         for i in 1..ind_r.len() - 1 {
//             let start_index = (ind_r[i] - 35) as usize;
//             let end_index = (ind_r[i] + 36) as usize;
//             let qrs1 = &mut ch1[start_index..end_index].to_vec();
//             let qrs2 = &mut ch2[start_index..end_index].to_vec();
//             let qrs3 = &mut ch3[start_index..end_index].to_vec();
//             let start_index = (ind_r[i + 1] - 35) as usize;
//             let end_index = (ind_r[i + 1] + 36) as usize;
//             let qrs11 = &mut ch1[start_index..end_index].to_vec();
//             let qrs22 = &mut ch2[start_index..end_index].to_vec();
//             let qrs33 = &mut ch3[start_index..end_index].to_vec();
//
//             let cor1 = self.get_coef_cor(qrs1, qrs11);
//             let cor2 = self.get_coef_cor(qrs2, qrs22);
//             let cor3 = self.get_coef_cor(qrs3, qrs33);
//
//             if cor1 > 0.93 && cor2 > 0.93 && cor3 > 0.93 {
//                 for i in 0..71 {
//                     self.ref_form1[i] = qrs1[i];
//                     self.ref_form2[i] = qrs2[i];
//                     self.ref_form3[i] = qrs3[i];
//                 }
//                 break;
//             }
//         }
//     }
//     fn get_ind_zero(&self, ind_forms: &Vec<u64>) -> Vec<u64> {
//         let mut out = vec![] as Vec<u64>;
//         for i in 0..ind_forms.len() {
//             if ind_forms[i] == 0 {
//                 out.push(i as u64);
//             }
//         }
//         out
//     }
//     fn get_rem_ind_r(&self, ind_r: &Vec<u64>, ind_rem: &Vec<u64>) -> Vec<u64> {
//         let mut out = vec![] as Vec<u64>;
//         for i in 0..ind_rem.len() {
//             out.push(ind_r[ind_rem[i] as usize]);
//         }
//         out


    // fn get_ind_types(&mut self, ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>, ind_r: &Vec<u64>) {
    //     let mut ind_forms:Vec<u64> = vec![0; ind_r.len()];
    //     for k in 1..100 {
    //         let ind_rem = self.get_ind_zero(&mut ind_forms);
    //         let rem_ind_r = self.get_rem_ind_r(&ind_r, &ind_rem);
    //         let _ = &self.get_ref_forms(&ch1, &ch2, &ch3, &rem_ind_r);
    //         for i in 0..ind_rem.len() {
    //             let mut coef_cor1 = vec![0.0; 9];
    //             let mut coef_cor2 = vec![0.0; 9];
    //             let mut coef_cor3 = vec![0.0; 9];
    //             for j in 0..9 {
    //                 let qrs1 = ch1[ind_r[i] as usize - 35 + j - 4..ind_r[i] as usize + 36 + j - 4].to_vec();
    //                 // let mut ref_form1 = &self.ref_form1;
    //                 coef_cor1[j] = self.get_coef_cor(&qrs1, &self.ref_form1);
    //             }
    //         }
    //     }
    // }
// }

fn main() {
    let mut ecg = ecg_data::Ecg {
        lead1: vec![],
        lead2: vec![],
        lead3: vec![],
        len_lead: 0,
        ind_r: vec![],
        intervals_r: vec![],
        div_intervals: vec![],
    };

    // let mut file1 = File::create("ch1.bin").expect("Не удалось создать файл");
    // let mut file2 = File::create("ch2.bin").expect("Не удалось создать файл");
    // let mut file3 = File::create("ch3.bin").expect("Не удалось создать файл");
    // let mut filef1 = File::create("fch1.bin").expect("Не удалось создать файл");
    // let mut filef2 = File::create("fch2.bin").expect("Не удалось создать файл");
    // let mut filef3 = File::create("fch3.bin").expect("Не удалось создать файл");

    ecg.read_ecg();

    // let slice1: &[f32] = &ecg.lead1;
    // let slice2: &[f32] = &ecg.lead2;
    // let slice3: &[f32] = &ecg.lead3;
    //
    // file1
    //     .write_all(bytemuck::cast_slice(slice1))
    //     .expect("Не удалось записать в файл");
    // file2
    //     .write_all(bytemuck::cast_slice(slice2))
    //     .expect("Не удалось записать в файл");
    // file3
    //     .write_all(bytemuck::cast_slice(slice3))
    //     .expect("Не удалось записать в файл");

    // let sum_leads = proc_ecg::pre_proc_r(&ecg.lead1, &ecg.lead2, &ecg.lead3);
    // let _ = ecg.get_ind_r(&sum_leads);
    // let _ = ecg.get_div_intervals();

    // let mut forms = FormsQrs {
    //     ref_form1: vec![0.0; 71],
    //     ref_form2: vec![0.0; 71],
    //     ref_form3: vec![0.0; 71],
    // };

    // let _ = forms.get_ref_forms(&ecg.lead1, &ecg.lead2, &ecg.lead3, &ecg.ind_r);

    // forms.ref_form1 = forms.norm_qrs(&forms.ref_form1.to_owned());
    // forms.ref_form2 = forms.norm_qrs(&forms.ref_form2.to_owned());
    // forms.ref_form3 = forms.norm_qrs(&forms.ref_form3.to_owned());

    // let cor = forms.get_coef_cor(&mut forms.ref_form1.to_owned(), &mut forms.ref_form3.to_owned());
    // println!("{}", cor);

    // let slicef3: &[f32] = &sum_leads;

    // filef3
    //     .write_all(bytemuck::cast_slice(slicef3))
    //     .expect("Не удалось записать в файл");
}
