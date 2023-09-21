use bytemuck;
use std::fs::File;
use std::io::Read;
use std::io::Write;


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

    for i in (3..lench - 3).step_by(1) {
        let j = i % 11;
        win[j] = ch[i];

        if (out[i] - out[i - 1]).abs() > 2.0 {
            // 0.3
            let mut sort_win = win.to_vec();
            sort_win.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out[i - 3] = sort_win[5];
            out[i - 2] = sort_win[5];
            out[i - 1] = sort_win[5];
            out[i] = sort_win[5];
            out[i + 1] = sort_win[5];
            out[i + 2] = sort_win[5];
            out[i + 3] = sort_win[5];
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

fn clean_ch(b_peak24: &Vec<f32>, a_peak24: &Vec<f32>,
            bl24: &Vec<f32>, al24: &Vec<f32>,
            bh24: &Vec<f32>, ah24: &Vec<f32>,
            b_peak50: &Vec<f32>, a_peak50: &Vec<f32>,
            bl50: &Vec<f32>, al50: &Vec<f32>,
            b: &Vec<f32>, a: &Vec<f32>,
            ch: &Vec<f32>) -> Vec<f32> {
    let ch_del_ks = cut_impuls(&ch);
    let spec24 = my_filtfilt(&b_peak24, &a_peak24, &ch_del_ks);
    let spec24 = spec24.iter().map(|&x| x.abs()).collect::<Vec<f32>>();
    let spec24 = my_filtfilt(&bl24, &al24, &spec24);
    let spec24 = my_filtfilt(&bh24, &ah24, &spec24);
    let spec24 = spec24.iter().map(|&x| x * 4.0).collect::<Vec<f32>>();
    let spec50 = my_filtfilt(&b_peak50, &a_peak50, &ch_del_ks);
    let spec50 = spec50.iter().map(|&x| x.abs()).collect::<Vec<f32>>();
    let spec50 = my_filtfilt(&bl50, &al50, &spec50);

    let clean_ch = my_filtfilt(&b, &a, &ch_del_ks);

    let mut fch = ch_del_ks.clone();

    for i in 0..fch.len() {
        if spec50[i] > spec24[i] {
            fch[i] = clean_ch[i];
        }
    }
    fch
}

fn main() {
    // bhn, ahn = butter(2, 6, 'hp', fs=250)
    let bhn = vec![0.89884553, -1.79769105, 0.89884553];
    let ahn = vec![1.0, -1.78743252, 0.80794959];

    // bln, aln = butter(6, 25, 'lp', fs=250)
    let bln = vec![0.00034054, 0.00204323, 0.00510806, 0.00681075, 0.00510806, 0.00204323, 0.00034054];
    let aln = vec![1.0, -3.5794348, 5.65866717, -4.96541523, 2.52949491, -0.70527411, 0.08375648];

    // bi, ai = butter(2, 0.6, 'hp', fs=250)
    let bi = vec![0.98939373, -1.97878745, 0.98939373];
    let ai = vec![1.0, -1.97867496, 0.97889995];

    // blr, alr = butter(1, 0.15, 'lp', fs=250)
    let blr = vec![0.00188141, 0.00188141];
    let alr = vec![1.0, -0.99623718];

    // bhr, ahr = butter(1, 6.1, 'hp', fs=250)
    let bhr = vec![0.92867294, -0.92867294];
    let ahr = vec![1.0, -0.85734589];

    // b_peak24, a_peak24 = iirpeak(24, 5, fs=250)
    let b_peak24 = vec![0.05695238, 0.0, -0.05695238];
    let a_peak24 = vec![1.0, -1.55326091, 0.88609524];

    // b_peak50, a_vec!iirpeak(50, 4, fs=250)
    let b_peak50 = vec![0.13672874, 0.0, -0.13672874];
    let a_peak50 = vec![1.0, -0.53353098, 0.72654253];

    // bl24, al24 = butter(1, 10.0, 'lp', fs=250)
    let bl24 = vec![0.11216024, 0.11216024];
    let al24 = vec![1.0, -0.77567951];

    // bh24, ahvec!er(1, 2, 'hp', fs=250)
    let bh24 = vec![0.97547839, -0.97547839];
    let ah24 = vec![1.0, -0.95095678];

    // bl50, alvec!er(1, 25.0, 'lp', fs=250)
    let bl50 = vec![0.24523728, 0.24523728];
    let al50 = vec![1.0, -0.50952545];

    // b, a = butter(2, 20, 'lp', fs=250)
    let b = vec![0.0461318, 0.0922636, 0.0461318];
    let a = vec![1.0, -1.30728503, 0.49181224];

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

    let bln = vec![0.00034054, 0.00204323, 0.00510806, 0.00681075, 0.00510806, 0.00204323, 0.00034054];
    let aln = vec![1.0, -3.5794348, 5.65866717, -4.96541523, 2.52949491, -0.70527411, 0.08375648];

    // let fch1 = my_filtfilt(&bln, &aln, &mut ch1);
    let fch1 = clean_ch(&b_peak24, &a_peak24, &bl24, &al24, &bh24, &ah24,
                        &b_peak50, &a_peak50, &bl50, &al50, &b, &a, &ch1);
    let fch2 = clean_ch(&b_peak24, &a_peak24, &bl24, &al24, &bh24, &ah24,
                        &b_peak50, &a_peak50, &bl50, &al50, &b, &a, &ch2);
    let fch3 = clean_ch(&b_peak24, &a_peak24, &bl24, &al24, &bh24, &ah24,
                        &b_peak50, &a_peak50, &bl50, &al50, &b, &a, &ch3);

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
