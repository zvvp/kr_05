use bytemuck;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::time::Instant;
use ndarray::prelude::*;
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
        pub ind_r: Vec<usize>,
        pub intervals_r: Vec<usize>,
        pub div_intervals: Vec<Option<f32>>,
    }

    impl Ecg {
        pub fn read_ecg(&mut self) {
            let files = glob::glob("*.ecg").expect("Failed to read files");
            let fname = files.filter_map(Result::ok).next().unwrap();
            println!("{:?}", fname);

            let mut file = File::open(fname).expect("Failed to open file");
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).expect("Failed to read file");

            let ecg: Vec<i16> = buffer[1024..9_000_000]
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
            self.get_ind_r(&sum_leads.0);
            self.get_div_intervals();
        }
        fn get_ind_r(&mut self, ch: &Vec<f32>) {
            let mut max_val: f32 = 0.0;
            let mut ind_max = 0;
            let mut interval = 0;
            for (ind, val) in ch.iter().enumerate() {
                if *val > max_val {
                    max_val = *val;
                    ind_max = ind;
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
            while self.ind_r[0] < 55 {
                self.ind_r.remove(0);
                self.intervals_r.remove(0);

            }
            while (ch.len() - self.ind_r[self.ind_r.len() - 1]) < 55 {
                self.ind_r.remove(self.ind_r.len() - 1);
                self.intervals_r.remove(self.ind_r.len() - 1);
            }
            if (ch.len() - self.ind_r[self.ind_r.len() - 1]) < 40 {
                self.ind_r.remove(self.ind_r.len() - 1);
                self.intervals_r.remove(self.ind_r.len() - 1);
            }
        }
        fn get_div_intervals(&mut self) {
            if self.intervals_r.len() > 0 {
                let mut intervals = self.intervals_r.to_owned();
                intervals.push(intervals[intervals.len() - 1]);
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
    fn cut_impuls1(ch: &Vec<f32>) -> Vec<f32> {
        let mut out = ch.clone();
        let mean_d = get_mean_diff(&ch);
        for i in 5..ch.len() - 6 {
            let d_out0 = (&out[i] - &out[i - 1]) / &mean_d;
            let d_out1 = (&out[i + 1] - &out[i]) / &mean_d;
            let d_out2 = (&out[i + 2] - &out[i + 1]) / &mean_d;
            let sign0 = d_out0.signum();
            let sign1 = d_out1.signum();
            let sign2 = d_out2.signum();
            let abs0 = d_out0.abs();
            let abs1 = d_out1.abs();
            let abs2 = d_out2.abs();
            if (sign0 != sign1) && (sign1 != sign2) && (abs1 > 1.4)
                && (((abs0 - abs1).abs() < abs1 * 0.88) || ((abs1 - abs2).abs() < abs1 * 0.88)) {
                let win: Vec<&f32> = ch.iter().skip(i-5).take(11).collect();
                let mut sort_win = win.to_owned();
                sort_win.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[i - 2] = *sort_win[5];
                out[i - 1] = *sort_win[5];
                out[i] = *sort_win[5];
                out[i + 1] = *sort_win[5];
                out[i + 2] = *sort_win[5];
                out[i + 3] = *sort_win[5];
                if ((&out[i + 4] - &out[i + 3]) / &mean_d).abs() > 0.15 {
                    out[i + 4] = *sort_win[5];
                    out[i + 5] = *sort_win[5];
                }
            }
            if (sign0 != sign1) && (abs0 > 1.0) && (abs1 > 1.0) && ((abs0 - abs1).abs() < (abs0 + abs1) * 0.5) {
                let win: Vec<&f32> = ch.iter().skip(i-5).take(11).collect();
                let mut sort_win = win.to_owned();
                sort_win.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[i - 2] = *sort_win[5];
                out[i - 1] = *sort_win[5];
                out[i] = *sort_win[5];
                out[i + 1] = *sort_win[5];
                out[i + 2] = *sort_win[5];
                if ((&out[i + 2] - &out[i + 1]) / &mean_d).abs() > 0.05 {
                    out[i + 2] = *sort_win[5];
                    out[i + 3] = *sort_win[5];
                }
            }
        }
        out
    }

    fn cut_impuls(ch: &Vec<f32>) -> Vec<f32> {
        let mut out = ch.clone();
        let mut win = [0.0; 11];
        let lench = ch.len();
        let diff_ch = my_diff(&ch);
        let abs_diff: Vec<f32> = diff_ch.iter().map(|&x| x.abs()).collect();
        let sum: f32 = abs_diff.iter().sum();
        let mean_diff = sum / abs_diff.len() as f32;
        let trs: f32 = get_trs(&abs_diff);
        let mdch = get_mean_diff(&ch);
        dbg!(trs, mdch);
        for i in 3..lench - 5 {
            let j = i % 11;
            win[j] = ch[i];
            let d1 = ch[i - 1] - ch[i - 2];
            let d2 = ch[i] - ch[i - 1];
            // if i < 100 {dbg!(d2);}
            let d3 = ch[i + 1] - ch[i];
            if abs_diff[i] > trs / mean_diff * 0.45 { // 2.00
                // if (ch[i] - out[i - 1]).abs() > trs * 7.0 { // 2.00
                let mut sort_win = win.to_vec();
                sort_win.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[i - 2] = sort_win[5];
                out[i - 1] = sort_win[5];
                out[i] = sort_win[5];
                out[i + 1] = sort_win[5];
                out[i + 2] = sort_win[5];
                out[i + 3] = sort_win[5];
                // out[i + 4] = sort_win[5];
                // out[i + 5] = sort_win[5];
            }
            // dbg!(d2.abs() + d3.abs());
            // dbg!(trs*0.5);
            if (d1.signum() != d2.signum()) && (d2.signum() != d3.signum()) && ((d1.abs() + d2.abs() + d3.abs()) > trs / mean_diff * 1.2) {  // 2.0
                let mut sort_win = win.to_vec();
                sort_win.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[i - 2] = sort_win[5];
                out[i - 1] = sort_win[5];
                out[i] = sort_win[5];
                out[i + 1] = sort_win[5];
                out[i + 2] = sort_win[5];
                out[i + 3] = sort_win[5];
            }
            // if (d2.signum() != d3.signum()) && ((d2 + d3).abs() < trs * 6.0) && ((d2.abs() + d3.abs()) > trs * 6.0) {  // 2.2, 0.2 0.4
            //     let mut sort_win = win.to_vec();
            //     sort_win.sort_by(|a, b| a.partial_cmp(b).unwrap());
            //     out[i - 2] = sort_win[5];
            //     out[i - 1] = sort_win[5];
            //     out[i] = sort_win[5];
            //     out[i + 1] = sort_win[5];
            //     out[i + 2] = sort_win[5];
            //     out[i + 3] = sort_win[5];
            // }
        }
        out
    }

    pub fn my_filtfilt(b: &Vec<f32>, a: &Vec<f32>, ch: &Vec<f32>) -> Vec<f32> {
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

    pub fn clean_ch(ch: &Vec<f32>) -> Vec<f32> {
        // b, a = butter(2, 20, 'lp', fs=250)
        let b = vec![0.0461318, 0.0922636, 0.0461318];
        let a = vec![1.0, -1.30728503, 0.49181224];
        // b, a = butter(2, 12, 'lp', fs=250)
        // let b = vec![0.0186504, 0.03730079, 0.0186504];
        // let a = vec![1.0, -1.57823618, 0.65283776];
        let ch_del_ks = cut_impuls1(&ch);
        let ch_del_ks = cut_impuls1(&ch_del_ks);
        let ch_del_ks = cut_impuls1(&ch_del_ks);
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

    pub fn get_p2p(ch: &Vec<f32>, win: usize, sqr: bool) -> Vec<f32> {
        let half_win = win / 2;
        let len_ch = ch.len();
        let mut p2p = vec![0.0; len_ch];
        for i in (0..len_ch - &win).step_by(2) {
            let win_ch = &ch[i..i + &win];
            let win_max = win_ch.iter().fold(f32::NEG_INFINITY, |max, &x| x.max(max));
            let win_min = win_ch.iter().fold(f32::INFINITY, |min, &x| x.min(min));
            let mut p2pw = &win_max - &win_min;
            if sqr {
                p2pw = &p2pw * &p2pw;
                // if p2pw > 2.0 {p2pw = 2.0 + p2pw * 0.2};
            }
            p2p[i + half_win - 2] = p2pw;
            p2p[i + half_win - 1] = p2pw;
            p2p[i + half_win] = p2pw;
            p2p[i + half_win + 1] = p2pw;
            p2p[i + half_win + 2] = p2pw;
        }
        if sqr {
            let trs = get_trs(&p2p);
            // dbg!(trs);
            if trs > 0.0 {
                for i in 0..p2p.len() {
                    p2p[i] /= trs;
                    if p2p[i] > 0.5 { p2p[i] = 0.5 + p2p[i] * 0.1; }
                    // if p2p[i] > trs * 1.5 { p2p[i] = trs * 1.5 + 0.2 * p2p[i]; }
                }
            }
        }
        p2p
    }

    fn sign(x: &f32) -> f32 {
        if *x > 0.0 {
            1.0
        } else if *x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    fn vec_sign(ch: &Vec<f32>) -> Vec<f32> {
        let out = ch.iter().map(|&val| sign(&val)).collect();
        out
    }

    fn my_diff(ch: &Vec<f32>) -> Vec<f32> {
        let ch_copy = ch.to_owned();
        let mut ch_copy1 = ch.to_owned();
        ch_copy1.remove(0);
        ch_copy1.push(0.0);
        let diff_ch: Vec<f32> = ch_copy.iter()
            .zip(ch_copy1.iter())
            .map(|(val1, val2)| val1 - val2).collect();
        diff_ch
    }

    pub fn get_mean_diff(ch: &Vec<f32>) -> f32 {
        let d_ch = my_diff(&ch);
        let abs_d_ch: Vec<f32> = d_ch.iter().map(|val| val.abs()).collect();
        let sum_d_ch: f32 = abs_d_ch.iter().sum();
        let mean_d_ch = sum_d_ch / d_ch.len() as f32;
        let (count, sum): (usize, f32) = abs_d_ch
            .iter()
            .filter(|&x| *x > mean_d_ch)
            .fold((0, 0.0), |(count, sum), &x| (count + 1, sum + x));
        let mean_d_ch1 = sum / count as f32;
        let (count, sum): (usize, f32) = abs_d_ch
            .iter()
            .filter(|&x| *x > mean_d_ch1)
            .fold((0, 0.0), |(count, sum), &x| (count + 1, sum + x));
        let mean_d_ch2 = sum / count as f32;
        let out = mean_d_ch + mean_d_ch1 + mean_d_ch2;
        out
    }

    pub fn get_trs(p2p: &Vec<f32>) -> f32 {
        let sum_p2p: f32 = p2p
            .iter()
            .sum();
        let mean_p2p: f32 = sum_p2p / p2p.len() as f32;
        let (count, sum): (usize, f32) = p2p
            .iter()
            .filter(|&x| *x > mean_p2p)
            .fold((0, 0.0), |(count, sum), &x| (count + 1, sum + x));
        let mut trs = sum / count as f32;
        trs
    }

    pub fn del_artifacts(ch: &Vec<f32>, p2p: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let len_ch = ch.len();
        let mut out = ch.to_owned();
        let mut mask = vec![1.0; len_ch];

        let signs = vec_sign(&ch);
        let diff_signs: Vec<f32> = my_diff(&signs);

        let mut prev_ind = 0;
        let mut flag: bool = false;
        let trs = get_trs(&p2p);

        for i in 0..diff_signs.len() - 1 {
            if (p2p[i] > trs * 6.0) || p2p[i] > 5.5 {                 // 3.8 2.6
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
        // bhn, ahn = butter(2, 6, 'hp', fs=250)
        let bhn = vec![0.89884553, -1.79769105, 0.89884553];
        let ahn = vec![1.0, -1.78743252, 0.80794959];
        // bln, aln = butter(6, 25, 'lp', fs=250)
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

    pub fn filt_r(ch: &Vec<f32>) -> Vec<f32> {
        //blr, alr = butter(1, 0.15, 'lp', fs=250)
        let blr = vec![0.00188141, 0.00188141];
        let alr = vec![1.0, -0.99623718];
        //blr, alr = butter(1, 0.1, 'lp', fs=250)
        // let blr = vec![0.00125506, 0.00125506];
        // let alr = vec![1.0, -0.99748988];
        // bhr, ahr = butter(1, 6.1, 'hp', fs=250)
        let bhr = vec![0.92867294, -0.92867294];
        let ahr = vec![1.0, -0.85734589];
        //bhr, ahr = butter(1, 5.5, 'hp', fs=250)
        // let bhr = vec![0.9521018, -0.9521018];
        // let ahr = vec![1.0, -0.90420359];

        let out = my_filtfilt(&blr, &alr, &ch);
        let out = out
            .iter()
            .map(|&x| x * 2000.0)
            .collect();
        let out = my_filtfilt(&bhr, &ahr, &out);
        out
    }

    fn sum_ch(ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>) -> Vec<f32> {
        let mut sum_ch: Vec<f32> = ch1.to_owned();
        for i in 0..ch1.len() - 1 {
            if ((ch1[i] > ch2[i]) && (ch1[i] < ch3[i])) || ((ch1[i] > ch3[i]) && (ch1[i] < ch2[i])) {
                sum_ch[i] = ch1[i];
            }
            if ((ch2[i] > ch1[i]) && (ch2[i] < ch3[i])) || ((ch2[i] > ch3[i]) && (ch2[i] < ch1[i])) {
                sum_ch[i] = ch2[i];
            }
            if ((ch3[i] > ch1[i]) && (ch3[i] < ch2[i])) || ((ch3[i] > ch2[i]) && (ch3[i] < ch1[i])) {
                sum_ch[i] = ch3[i];
            }
        }
        sum_ch
    }
    // fn sum_ch(ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>) -> Vec<f32> {
    //     let mut sum_ch: Vec<f32> = ch1
    //         .iter()
    //         .zip(ch2.iter())
    //         .zip(ch3.iter())
    //         .map(|((&a, &b), &c)| {
    //             let sum = a + b + c;
    //             sum
    //         })
    //         .collect();
    //     // let trs = get_trs(&sum_ch) * 0.8;
    //     // for i in 0..sum_ch.len() {
    //     //     if sum_ch[i] > trs {
    //     //         sum_ch[i] = trs + sum_ch[i] * 0.1;
    //     //     }
    //     // }
    //     sum_ch
    // }

    pub fn pre_proc_r(ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let cln_ch1 = clean_ch(&ch1);
        let cln_ch2 = clean_ch(&ch2);
        let cln_ch3 = clean_ch(&ch3);

        let p2p_ch1 = get_p2p(&cln_ch1, 40, false);
        let p2p_ch2 = get_p2p(&cln_ch2, 40, false);
        let p2p_ch3 = get_p2p(&cln_ch3, 40, false);

        // let p2p_ch1 = del_isoline(&p2p_ch1);
        // let p2p_ch2 = del_isoline(&p2p_ch2);
        // let p2p_ch3 = del_isoline(&p2p_ch3);

        let art1 = del_artifacts(&cln_ch1, &p2p_ch1);
        let art2 = del_artifacts(&cln_ch2, &p2p_ch2);
        let art3 = del_artifacts(&cln_ch3, &p2p_ch3);

        let cln_ch1 = del_isoline(&art1.0);
        let cln_ch2 = del_isoline(&art2.0);
        let cln_ch3 = del_isoline(&art3.0);

        let fch1 = del_nouse(&cln_ch1, &art1.1);
        let fch2 = del_nouse(&cln_ch2, &art2.1);
        let fch3 = del_nouse(&cln_ch3, &art3.1);

        let fch1 = get_p2p(&fch1, 30, true);
        let fch2 = get_p2p(&fch2, 30, true);
        let fch3 = get_p2p(&fch3, 30, true);

        let fch1 = filt_r(&fch1);
        let fch2 = filt_r(&fch2);
        let fch3 = filt_r(&fch3);
        let sum_leads = sum_ch(&fch1, &fch2, &fch3);
        // let sum_leads = filt_r(&sum_leads);
        // (sum_leads, cln_ch1, cln_ch2, cln_ch3)
        (sum_leads, fch1, fch2, fch3)
    }

    pub fn del_isoline(ch: &Vec<f32>) -> Vec<f32> {
        let len_win = 120;
        // let ind_med = 60;
        let mut out = ch.to_owned();
        let mut isoline = ch.to_owned();
        let mut buff = vec![0.0; len_win];
        for i in 63..isoline.len() {
            let j = i % len_win;
            buff[j] = ch[i];
            let mut sort_buff: Vec<_> = buff.to_vec().iter().step_by(7).cloned().collect();
            sort_buff.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out[i - 60 - 3] = ch[i - 60 - 3] - sort_buff[8];
            out[i - 60 - 2] = ch[i - 60 - 2] - sort_buff[8];
            out[i - 60 - 1] = ch[i - 60 - 1] - sort_buff[8];
            out[i - 60] = ch[i - 60] - sort_buff[8];
            out[i - 60 + 1] = ch[i - 60 + 1] - sort_buff[8];
            out[i - 60 + 2] = ch[i - 60 + 2] - sort_buff[8];
            out[i - 60 + 3] = ch[i - 60 + 3] - sort_buff[8];
        }
        out
    }

    // pub fn del_isoline(ch: &Vec<f32>) -> Vec<f32> {
    //     // bi, ai = butter(2, 0.6, 'hp', fs=250)
    //     let bi = vec![0.98939373, -1.97878745, 0.98939373];
    //     let ai = vec![1.0, -1.97867496, 0.97889995];
    //     let out = my_filtfilt(&bi, &ai, &ch);
    //     out
    // }
}

mod qrs_forms {
    use ndarray::Array1;
    // use crate::proc_ecg::del_isoline;
    use crate::qrs_types::max_vec;

    pub struct FormsQrs {
        pub ref_form1: Vec<f32>,
        pub ref_form2: Vec<f32>,
        pub ref_form3: Vec<f32>,
    }

    impl FormsQrs {
        pub fn get_ref_forms(&mut self, ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>, ind_r: &Vec<usize>) {
            if ind_r.len() > 1 {
                for i in 0..(ind_r.len() - 1) {
                    let start_index = (ind_r[i] - 25) as usize;
                    let end_index = (ind_r[i] + 26) as usize;
                    let start_index1 = (ind_r[i + 1] - 25) as usize;
                    let end_index1 = (ind_r[i + 1] + 26) as usize;
                    let qrs1 = &ch1[start_index..end_index].to_vec();
                    let qrs2 = &ch2[start_index..end_index].to_vec();
                    let qrs3 = &ch3[start_index..end_index].to_vec();

                    let mut coef_cor1 = vec![0.0; 41];
                    let mut coef_cor2 = vec![0.0; 41];
                    let mut coef_cor3 = vec![0.0; 41];
                    for j in 0..41 {
                        let qrs11 = &ch1[start_index1 + j - 20..end_index1 + j - 20].to_vec();
                        coef_cor1[j] = get_coef_cor(&qrs1, &qrs11);
                        let qrs22 = &ch2[start_index1 + j - 20..end_index1 + j - 20].to_vec();
                        coef_cor2[j] = get_coef_cor(&qrs2, &qrs22);
                        let qrs33 = &ch3[start_index1 + j - 20..end_index1 + j - 20].to_vec();
                        coef_cor3[j] = get_coef_cor(&qrs3, &qrs33);
                    }
                    let max_cor1 = max_vec(&coef_cor1);
                    let max_cor2 = max_vec(&coef_cor2);
                    let max_cor3 = max_vec(&coef_cor3);

                    if max_cor1 > 0.93 && max_cor2 > 0.93 && max_cor3 > 0.93 {
                        self.ref_form1 = qrs1.to_owned();
                        self.ref_form2 = qrs2.to_owned();
                        self.ref_form3 = qrs3.to_owned();
                        break;
                    }
                    // let start_index = (ind_r[i + 1] - 35) as usize;
                    // let end_index = (ind_r[i + 1] + 36) as usize;
                    // let qrs11 = &ch1[start_index..end_index].to_vec();
                    // let qrs22 = &ch2[start_index..end_index].to_vec();
                    // let qrs33 = &ch3[start_index..end_index].to_vec();
                    //
                    // let cor1 = get_coef_cor(&qrs1, &qrs11);
                    // let cor2 = get_coef_cor(&qrs2, &qrs22);
                    // let cor3 = get_coef_cor(&qrs3, &qrs33);
                    // if cor1 > 0.97 && cor2 > 0.97 && cor3 > 0.97 {
                    //     // if cor1 > 0.925 && cor2 > 0.927 && cor3 > 0.926 {
                    //     self.ref_form1 = qrs1.to_owned();
                    //     self.ref_form2 = qrs2.to_owned();
                    //     self.ref_form3 = qrs3.to_owned();
                    //     break;
                }
            }
        }
    }

    pub fn norm_qrs(qrs: &Vec<f32>) -> Vec<f32> {
        let mut min: f32 = 0.0;
        let mut max: f32 = 0.0;
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

    pub fn get_coef_cor(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        let norm_x = norm_qrs(&x);
        let norm_y = norm_qrs(&y);
        let arr_x = Array1::from_vec(norm_x);
        let arr_y = Array1::from_vec(norm_y);
        let mean_x = &arr_x.mean().unwrap();
        let mean_y = &arr_y.mean().unwrap();
        let arr_xy = &arr_x * &arr_y;
        let mean_xy = Array1::from(arr_xy).mean().unwrap();
        let std_x = &arr_x.std(0.0);
        let std_y = &arr_y.std(0.0);
        let std_xy = std_x * std_y;
        let mut out: f32 = 0.0;
        if std_xy != 0.0 {
            out = (mean_xy - mean_x * mean_y) / std_xy;
        } else { out = 0.0; }
        out
    }

    pub fn median_cor(cor1: f32, cor2: f32, cor3: f32) -> f32 {
        let mut vec_cor = vec![cor1, cor2, cor3];
        vec_cor.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vec_cor[1]
    }

    pub fn max_cor(cor1: f32, cor2: f32, cor3: f32) -> f32 {
        let mut vec_cor = vec![cor1, cor2, cor3];
        vec_cor.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vec_cor[2]
    }
}

pub mod qrs_types {
    use crate::proc_ecg::*;
    use crate::qrs_forms::*;

    fn get_ind_zero(ind_forms: &Vec<usize>) -> Vec<usize> {
        let mut out = vec![];
        for i in 0..ind_forms.len() {
            if ind_forms[i] == 0 {
                out.push(i);
            }
        }
        out
    }

    fn get_rem_ind_r(ind_r: &Vec<usize>, ind_rem: &Vec<usize>) -> Vec<usize> {
        let mut out = vec![];
        for i in 0..ind_rem.len() {
            out.push(ind_r[ind_rem[i]]);
        }
        out
    }

    pub fn max_vec(vec: &Vec<f32>) -> f32 {
        let mut max_v = 0.0;
        for v in vec.iter() {
            if *v > max_v { max_v = *v };
        }
        max_v
    }

    pub fn get_ind_types(ch1: &Vec<f32>, ch2: &Vec<f32>, ch3: &Vec<f32>, ind_r: &Vec<usize>) -> Vec<usize> {
        let mut ind_forms: Vec<usize> = vec![0; ind_r.len()];
        let mut forms = FormsQrs {
            ref_form1: vec![0.0; 51],
            ref_form2: vec![0.0; 51],
            ref_form3: vec![0.0; 51],
        };
        let fch1 = clean_ch(&ch1);
        let fch2 = clean_ch(&ch2);
        let fch3 = clean_ch(&ch3);

        let fch1 = del_isoline(&fch1);
        let fch2 = del_isoline(&fch2);
        let fch3 = del_isoline(&fch3);

        let mut ind_rem_size = 0;
        for k in 1..10 {
            let ind_rem = get_ind_zero(&mut ind_forms);
            ind_rem_size = ind_rem.len();
            println!("{}", ind_rem_size);
            if ind_rem_size < 100 {break;}
            let rem_ind_r = get_rem_ind_r(&ind_r, &ind_rem);
            let _ = forms.get_ref_forms(&fch1, &fch2, &fch3, &rem_ind_r);

            for i in 0..ind_rem.len() {
                let mut coef_cor1 = vec![0.0; 41];
                let mut coef_cor2 = vec![0.0; 41];
                let mut coef_cor3 = vec![0.0; 41];
                for j in 0..41 {
                    let qrs1 = &fch1[ind_r[ind_rem[i]] - 25 + j - 20..ind_r[ind_rem[i]] + 26 + j - 20].to_vec();
                    coef_cor1[j] = get_coef_cor(&qrs1, &forms.ref_form1);
                    let qrs2 = &fch2[ind_r[ind_rem[i]] - 25 + j - 20..ind_r[ind_rem[i]] + 26 + j - 20].to_vec();
                    coef_cor2[j] = get_coef_cor(&qrs2, &forms.ref_form2);
                    let qrs3 = &fch3[ind_r[ind_rem[i]] - 25 + j - 20..ind_r[ind_rem[i]] + 26 + j - 20].to_vec();
                    coef_cor3[j] = get_coef_cor(&qrs3, &forms.ref_form3);
                }
                let max_cor1 = max_vec(&coef_cor1);
                let max_cor2 = max_vec(&coef_cor2);
                let max_cor3 = max_vec(&coef_cor3);

                if max_cor1 > 0.955 || max_cor2 > 0.955 || max_cor3 > 0.955 {
                    ind_forms[ind_rem[i]] = k;
                } else if max_cor1 > 0.84 && max_cor2 > 0.84 && max_cor3 > 0.84 {
                    ind_forms[ind_rem[i]] = k;
                }
            }
        }
        // println!("{}", ind_forms.len());
        ind_forms
    }
}

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

    let mut file1 = File::create("ch1.bin").expect("Не удалось создать файл");
    let mut file2 = File::create("ch2.bin").expect("Не удалось создать файл");
    let mut file3 = File::create("ch3.bin").expect("Не удалось создать файл");
    let mut filef1 = File::create("fch1.bin").expect("Не удалось создать файл");
    let mut filef2 = File::create("fch2.bin").expect("Не удалось создать файл");
    let mut filef3 = File::create("fch3.bin").expect("Не удалось создать файл");
    let mut filef4 = File::create("fch4.bin").expect("Не удалось создать файл");

    ecg.read_ecg();
    println!("{}", ecg.ind_r.len());
    // println!("{}", ecg.intervals_r.len());
    // println!("{}", ecg.div_intervals.len());
    // println!("{:?}", &ecg.ind_r[..5]);
    // println!("{:?}", &ecg.div_intervals[..5]);

    // let start = Instant::now();
    // ecg.lead1 = proc_ecg::del_isoline(&ecg.lead1);
    // ecg.lead2 = proc_ecg::del_isoline(&ecg.lead2);
    // ecg.lead3 = proc_ecg::del_isoline(&ecg.lead3);
    // let stop = start.elapsed();
    // println!("Время выполнения: {} s", stop.as_secs());

    // ecg.lead1 = proc_ecg::clean_ch(&ecg.lead1);
    // ecg.lead2 = proc_ecg::clean_ch(&ecg.lead2);
    // ecg.lead3 = proc_ecg::clean_ch(&ecg.lead3);

    let slice1: &[f32] = &ecg.lead1;
    let slice2: &[f32] = &ecg.lead2;
    let slice3: &[f32] = &ecg.lead3;

    file1
        .write_all(bytemuck::cast_slice(slice1))
        .expect("Не удалось записать в файл");
    file2
        .write_all(bytemuck::cast_slice(slice2))
        .expect("Не удалось записать в файл");
    file3
        .write_all(bytemuck::cast_slice(slice3))
        .expect("Не удалось записать в файл");

    // ecg.lead1 = clean_ch(&ecg.lead1);
    // ecg.lead2 = clean_ch(&ecg.lead2);
    // ecg.lead3 = clean_ch(&ecg.lead3);

    // let p2p3 = get_p2p(&ecg.lead3, 30, true);
    // let p2p3 = filt_r(&p2p3);
    // let del_art = del_artifacts(&ecg.lead3, &p2p3);
    let start = Instant::now();
    let type_forms = qrs_types::get_ind_types(&ecg.lead1, &ecg.lead2, &ecg.lead3, &ecg.ind_r);
    let stop = start.elapsed();
    println!("Время выполнения: {} ms", stop.as_millis());
    let mut types = vec![0.0; ecg.lead1.len()];
    for i in 0..type_forms.len() {
        types[ecg.ind_r[i]] = type_forms[i] as f32;
    }

    // let sum_leads = proc_ecg::pre_proc_r(&ecg.lead1, &ecg.lead2, &ecg.lead3);
    // let sum_leads= proc_ecg::filt_r(&sum_leads.0);
    // let slicef1: &[f32] = &sum_leads.1;
    // let slicef2: &[f32] = &sum_leads.2;
    // let slicef3: &[f32] = &sum_leads;
    // let slicef4: &[f32] = &sum_leads.0;
    // let slicef3: &[f32] = &p2p3;
    let slicef3: &[f32] = &types;
    let slicef4: &[f32] = &ecg.div_intervals.iter()
        .flat_map(|o| o.as_ref())
        .cloned()
        .collect::<Vec<f32>>();
    // let slicef3: &[usize] = &ecg.ind_r;
    // filef1
    //     .write_all(bytemuck::cast_slice(slicef1))
    //     .expect("Не удалось записать в файл");
    // filef2
    //     .write_all(bytemuck::cast_slice(slicef2))
    //     .expect("Не удалось записать в файл");
    filef3
        .write_all(bytemuck::cast_slice(slicef3))
        .expect("Не удалось записать в файл");
    filef4
        .write_all(bytemuck::cast_slice(slicef4))
        .expect("Не удалось записать в файл");
}
