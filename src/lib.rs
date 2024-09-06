#![feature(stdarch_x86_avx512)]

use pyo3::{ wrap_pyfunction, prelude::*, };
use numpy::{PyArray1, ndarray::ArrayView1, PyArrayMethods};


fn fast_standard_deviation_internal_manual_simd_avx512(array_view: ArrayView1<f64>) -> f64 {
    use core::arch::x86_64::*;

    // Number of elements
    let len = array_view.len();
    let len_f64 = array_view.len() as f64;

    // Sum the elements of the array
    let sum = unsafe {
        let mut ptr = array_view.as_ptr();
        let end = ptr.add(len);
        let mut sum_a: __m512d = _mm512_set1_pd(0.0);
        let mut sum_b: __m512d = _mm512_set1_pd(0.0);
        let mut sum_c: __m512d = _mm512_set1_pd(0.0);
        let mut sum_d: __m512d = _mm512_set1_pd(0.0);
        while ptr < end {
            sum_a = _mm512_add_pd(sum_a, _mm512_loadu_pd(ptr));
            sum_b = _mm512_add_pd(sum_b, _mm512_loadu_pd(ptr.offset(8)));
            sum_c = _mm512_add_pd(sum_c, _mm512_loadu_pd(ptr.offset(16)));
            sum_d = _mm512_add_pd(sum_d, _mm512_loadu_pd(ptr.offset(24)));
            ptr = ptr.offset(32);
        }
        _mm512_reduce_add_pd(sum_a) +
            _mm512_reduce_add_pd(sum_b) +
            _mm512_reduce_add_pd(sum_c) +
            _mm512_reduce_add_pd(sum_d)
    };

    // The mean
    let mean = sum / len_f64;

    let square_diff_sum = unsafe {
        let mean = _mm512_set1_pd(mean);
        let mut ptr = array_view.as_ptr();
        let end = ptr.add(len);
        let mut sum_a: __m512d = _mm512_set1_pd(0.0);
        let mut sum_b: __m512d = _mm512_set1_pd(0.0);
        let mut sum_c: __m512d = _mm512_set1_pd(0.0);
        let mut sum_d: __m512d = _mm512_set1_pd(0.0);
        let mut diff_a: __m512d;
        let mut diff_b: __m512d;
        let mut diff_c: __m512d;
        let mut diff_d: __m512d;
        while ptr < end {

            diff_a = _mm512_sub_pd(_mm512_loadu_pd(ptr), mean);
            sum_a = _mm512_add_pd(sum_a, _mm512_mul_pd(diff_a, diff_a));

            diff_b = _mm512_sub_pd(_mm512_loadu_pd(ptr.offset(8)), mean);
            sum_b = _mm512_add_pd(sum_b, _mm512_mul_pd(diff_b, diff_b));

            diff_c = _mm512_sub_pd(_mm512_loadu_pd(ptr.offset(16)), mean);
            sum_c = _mm512_add_pd(sum_c, _mm512_mul_pd(diff_c, diff_c));

            diff_d = _mm512_sub_pd(_mm512_loadu_pd(ptr.offset(24)), mean);
            sum_d = _mm512_add_pd(sum_d, _mm512_mul_pd(diff_d, diff_d));

            ptr = ptr.offset(32);
        }
        _mm512_reduce_add_pd(sum_a) +
            _mm512_reduce_add_pd(sum_b) +
            _mm512_reduce_add_pd(sum_c) +
            _mm512_reduce_add_pd(sum_d)
    };

    (square_diff_sum / len_f64).sqrt()
}

fn fast_standard_deviation_internal_manual_simd_avx2(array_view: ArrayView1<f64>) -> f64 {
    use core::arch::x86_64::*;

    // Number of elements
    let len = array_view.len();
    let len_f64 = array_view.len() as f64;

    // Sum the elements of the array
    let sum = unsafe {
        let mut ptr = array_view.as_ptr();
        let end = ptr.add(len);
        let mut sum_a: __m256d = _mm256_set1_pd(0.0);
        let mut sum_b: __m256d = _mm256_set1_pd(0.0);
        let mut sum_c: __m256d = _mm256_set1_pd(0.0);
        let mut sum_d: __m256d = _mm256_set1_pd(0.0);
        while ptr < end {
            sum_a = _mm256_add_pd(sum_a, _mm256_loadu_pd(ptr));
            sum_b = _mm256_add_pd(sum_b, _mm256_loadu_pd(ptr.offset(4)));
            sum_c = _mm256_add_pd(sum_c, _mm256_loadu_pd(ptr.offset(8)));
            sum_d = _mm256_add_pd(sum_d, _mm256_loadu_pd(ptr.offset(12)));
            ptr = ptr.offset(16);
        }
        sum_a = _mm256_add_pd(sum_a, sum_b);
        sum_c = _mm256_add_pd(sum_c, sum_d);

        sum_a = _mm256_add_pd(sum_a, sum_c);
        sum_a = _mm256_hadd_pd(sum_a, sum_a);
        sum_a = _mm256_hadd_pd(sum_a, sum_a);
        _mm256_cvtsd_f64(sum_a)
    };

    // The mean
    let mean = sum / len_f64;

    let square_diff_sum = unsafe {
        let mean = _mm256_set1_pd(mean);
        let mut ptr = array_view.as_ptr();
        let end = ptr.add(len);
        let mut sum_a: __m256d = _mm256_set1_pd(0.0);
        let mut sum_b: __m256d = _mm256_set1_pd(0.0);
        let mut sum_c: __m256d = _mm256_set1_pd(0.0);
        let mut sum_d: __m256d = _mm256_set1_pd(0.0);
        let mut diff_a: __m256d;
        let mut diff_b: __m256d;
        let mut diff_c: __m256d;
        let mut diff_d: __m256d;
        while ptr < end {

            diff_a = _mm256_sub_pd(_mm256_loadu_pd(ptr), mean);
            sum_a = _mm256_add_pd(sum_a, _mm256_mul_pd(diff_a, diff_a));

            diff_b = _mm256_sub_pd(_mm256_loadu_pd(ptr.offset(4)), mean);
            sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(diff_b, diff_b));

            diff_c = _mm256_sub_pd(_mm256_loadu_pd(ptr.offset(8)), mean);
            sum_c = _mm256_add_pd(sum_c, _mm256_mul_pd(diff_c, diff_c));

            diff_d = _mm256_sub_pd(_mm256_loadu_pd(ptr.offset(12)), mean);
            sum_d = _mm256_add_pd(sum_d, _mm256_mul_pd(diff_d, diff_d));

            ptr = ptr.offset(16);
        }
        sum_a = _mm256_add_pd(sum_a, sum_b);
        sum_c = _mm256_add_pd(sum_c, sum_d);

        sum_a = _mm256_add_pd(sum_a, sum_c);
        sum_a = _mm256_hadd_pd(sum_a, sum_a);
        sum_a = _mm256_hadd_pd(sum_a, sum_a);
        _mm256_cvtsd_f64(sum_a)
    };

    (square_diff_sum / len_f64).sqrt()
}

fn fast_standard_deviation_internal(array_view: ArrayView1<f64>) -> f64 {

    // Number of elements
    let len = array_view.len() as f64;

    // Sum the elements of the array
    let sum = array_view.iter().sum::<f64>();

    // The mean
    let mean = sum / len;

    let square_diff_sum = array_view.iter().map(|x| (x - mean).powi(2)).sum::<f64>();

    (square_diff_sum / len).sqrt()
}

#[pyfunction]
fn fast_standard_deviation_avx512(arr: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    // Create an ArrayView from the PyArray without copying
    let array_view: ArrayView1<f64> = unsafe { arr.as_array() };

    // Ok(fast_standard_deviation_internal_manual_simd(array_view))
    Ok(fast_standard_deviation_internal_manual_simd_avx512(array_view))
}

#[pyfunction]
fn fast_standard_deviation_avx2(arr: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    // Create an ArrayView from the PyArray without copying
    let array_view: ArrayView1<f64> = unsafe { arr.as_array() };

    // Ok(fast_standard_deviation_internal_manual_simd(array_view))
    Ok(fast_standard_deviation_internal_manual_simd_avx2(array_view))
}

#[pyfunction]
fn fast_standard_deviation(arr: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    // Create an ArrayView from the PyArray without copying
    let array_view: ArrayView1<f64> = unsafe { arr.as_array() };

    // Ok(fast_standard_deviation_internal_manual_simd(array_view))
    Ok(fast_standard_deviation_internal(array_view))
}

#[pymodule]
fn your_python_is_slow(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_standard_deviation, m)?)?;
    m.add_function(wrap_pyfunction!(fast_standard_deviation_avx2, m)?)?;
    m.add_function(wrap_pyfunction!(fast_standard_deviation_avx512, m)?)?;
    Ok(())
}


