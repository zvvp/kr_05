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

fn cut_impuls(ch: &mut Vec<f32>) -> Vec<f32> {
    let mut out = ch.clone();
    let mut win = [0.0; 11];
    let lench = ch.len();

    for i in (2..lench - 2).step_by(1) {
        let j = i % 11;
        win[j] = ch[i];

        if (out[i] - out[i - 1]).abs() > 0.3 {
            // 0.3
            let mut sort_win = win.to_vec();
            sort_win.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // out[i - 2] = sort_win[5];
            out[i - 1] = sort_win[5];
            out[i] = sort_win[5];
            out[i + 1] = sort_win[5];
            // out[i + 2] = sort_win[5];
        }
    }
    out
}



fn main() {
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

    let mut fch1 = cut_impuls(&mut ch1);
    let mut fch2 = cut_impuls(&mut ch2);
    let mut fch3 = cut_impuls(&mut ch3);

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
