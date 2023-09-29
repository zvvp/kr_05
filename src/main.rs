use bytemuck;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use ndarray::{Array, s};


fn read_ecg() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let files = glob::glob("*.ecg").expect("Failed to read files");
    let fname = files.filter_map(Result::ok).next().unwrap();
    println!("{:?}", fname);

    let mut file = File::open(fname).expect("Failed to open file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");

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

    (ch_1, ch_2, ch_3)
}

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
        let d4 = ch[i + 2] - ch[i + 1];
        let sum_d = d1.abs() + d2.abs() + d3.abs() + d4.abs();
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
    // ch_del_ks
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
            p2pw = &p2pw * &p2pw;
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
    let mut ch_copy = ch.to_owned();
    let mut ch_copy1 = ch.to_owned();
    ch_copy1.remove(0);
    ch_copy1.push(0.0);
    let diff_ch: Vec<f32> = ch_copy.iter().zip(ch_copy1.iter()).map(|(val1, val2)| val1 - val2).collect();
    diff_ch
}

fn get_trs(p2p:&Vec<f32>) -> f32 {
    let sum_p2p: f32 = p2p
        .iter()
        .sum();
    let mean_p2p: f32 = sum_p2p / p2p.len() as f32;
    let (count, sum): (usize, f32) = p2p
        .iter()
        .filter(|&x| *x > mean_p2p)
        .fold((0, 0.0), |(count, sum), &x| (count + 1, sum + x));
    let trs = sum / count as f32;
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
        if p2p[i] > trs * 3.0 {                 // 5.5
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

fn var_ch(ch: Vec<f32>, len_win: usize) -> Vec<f32> {
    let len_ch = ch.len();
    let nd_ch = Array::from_vec(ch);
    // let mut out = Array::zeros(len_ch);
    let mut out = vec![0.0; len_ch];
    let half_win = len_win / 2;
    for i in half_win..(len_ch - half_win) {
        let start = i - half_win;
        let end = i + half_win;
        let mut temp = nd_ch.slice(s![start..end]).var(0.0) * 10.0;
        if temp > 2.0 {
            temp = 2.0;
        }
        out[i] = temp;
    }
    out
}

fn main() {
    {
        // bi, ai = butter(2, 0.6, 'hp', fs=250)
        let bi = vec![0.98939373, -1.97878745, 0.98939373];
        let ai = vec![1.0, -1.97867496, 0.97889995];

        // blr, alr = butter(1, 0.15, 'lp', fs=250)
        let blr = vec![0.00188141, 0.00188141];
        let alr = vec![1.0, -0.99623718];

        // bhr, ahr = butter(1, 6.1, 'hp', fs=250)
        let bhr = vec![0.92867294, -0.92867294];
        let ahr = vec![1.0, -0.85734589];
    }
    let mut file1 = File::create("ch1.bin").expect("Не удалось создать файл");
    let mut file2 = File::create("ch2.bin").expect("Не удалось создать файл");
    let mut file3 = File::create("ch3.bin").expect("Не удалось создать файл");
    let mut filef1 = File::create("fch1.bin").expect("Не удалось создать файл");
    let mut filef2 = File::create("fch2.bin").expect("Не удалось создать файл");
    let mut filef3 = File::create("fch3.bin").expect("Не удалось создать файл");

    let ecg = read_ecg();
    let mut ch1 = ecg.0;
    let mut ch2 = ecg.1;
    let mut ch3 = ecg.2;

    let slice1: &[f32] = &ch1;
    let slice2: &[f32] = &ch2;
    let slice3: &[f32] = &ch3;

    file1
        .write_all(bytemuck::cast_slice(slice1))
        .expect("Не удалось записать в файл");
    file2
        .write_all(bytemuck::cast_slice(slice2))
        .expect("Не удалось записать в файл");
    file3
        .write_all(bytemuck::cast_slice(slice3))
        .expect("Не удалось записать в файл");

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

    // let fch1 = get_p2p(&fch1, 30, true);
    // let fch2 = get_p2p(&fch2, 30, true);
    // let fch3 = get_p2p(&fch3, 30, true);

    let slicef1: &[f32] = &fch1;
    let slicef2: &[f32] = &fch2;
    let slicef3: &[f32] = &fch3;

    filef1
        .write_all(bytemuck::cast_slice(slicef1))
        .expect("Не удалось записать в файл");
    filef2
        .write_all(bytemuck::cast_slice(slicef2))
        .expect("Не удалось записать в файл");
    filef3
        .write_all(bytemuck::cast_slice(slicef3))
        .expect("Не удалось записать в файл");
}
