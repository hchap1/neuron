#![allow(dead_code)]

use core::fmt;
use core::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use rand_distr::{Normal, Distribution};
use rand::rngs::ThreadRng;

#[derive(Copy, Clone, PartialEq)]
pub struct Matrix<const M: usize, const N: usize> {
    pub data: [[f64; N]; M],
}

impl<const M: usize, const N: usize> fmt::Debug for Matrix<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix<{}, {}> [", M, N)?;
        for (r, row) in self.data.iter().enumerate() {
            if r + 1 == M {
                writeln!(f, "  {:?}", row)?;
            } else {
                writeln!(f, "  {:?},", row)?;
            }
        }
        write!(f, "]")
    }
}

impl<const M: usize> Matrix<M, 1> {
    pub fn get(&self, index: usize) -> f64 {
        self.data[index][0]
    }

    pub fn set(&mut self, index: usize, value: f64) {
        self.data[index][0] = value
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {

    pub fn he_dist(rng: &mut ThreadRng) -> Self {
        let std_dev = (2.0 / (N as f64)).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        let mut matrix = Self::zeros();
        for row in 0..M {
            for col in 0..N {
                matrix.data[row][col] = normal.sample(rng);
            }
        }
        matrix
    }

    pub const fn zeros() -> Self {
        Self { data: [[0.0; N]; M] }
    }

    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        let mut out = *self;
        for i in 0..M {
            for j in 0..N {
                out.data[i][j] = f(self.data[i][j]);
            }
        }
        out
    }

    pub fn map_mut<F: Fn(f64) -> f64>(&mut self, f: F) {
        for i in 0..M {
            for j in 0..N {
                self.data[i][j] = f(self.data[i][j]);
            }
        }
    }

    pub fn hadamard(&self, other: &Self) -> Self {
        let mut out = *self;
        out.hadamard_mut(other);
        out
    }

    pub fn hadamard_mut(&mut self, other: &Self) {
        for i in 0..M {
            for j in 0..N {
                self.data[i][j] *= other.data[i][j];
            }
        }
    }

    pub fn transpose(&self) -> Matrix<N, M> {
        let mut out = Matrix::<N, M>::zeros();
        for i in 0..M {
            for j in 0..N {
                out.data[j][i] = self.data[i][j];
            }
        }
        out
    }

    pub fn argmax(&self) -> (usize, f64) {
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for i in 0..M {
            for j in 0..N {
                let idx = i * N + j;
                let v = self.data[i][j];
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
        }
        (best_idx, best_val)
    }

    pub fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(M * N);
        for i in 0..M {
            for j in 0..N {
                v.push(self.data[i][j]);
            }
        }
        v
    }

    pub fn from_flat(vec: Vec<f64>) -> Self {
        assert_eq!(vec.len(), M * N, "Invalid Size");
        let mut out = Self::zeros();
        for i in 0..M {
            for j in 0..N {
                out.data[i][j] = vec[i * N + j];
            }
        }
        out
    }

    pub fn from_2d(vv: Vec<Vec<f64>>) -> Result<Self, String> {
        if vv.len() != M {
            return Err(format!("from_2d: expected {} rows, got {}", M, vv.len()));
        }
        let mut out = Self::zeros();
        for (i, row) in vv.into_iter().enumerate() {
            if row.len() != N {
                return Err(format!(
                    "from_2d: row {} expected {} cols, got {}",
                    i, N, row.len()
                ));
            }
            for j in 0..N {
                out.data[i][j] = row[j];
            }
        }
        Ok(out)
    }
}

impl<const M: usize, const N: usize> Add for Matrix<M, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out += rhs;
        out
    }
}
impl<const M: usize, const N: usize> AddAssign for Matrix<M, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self.data[i][j] += rhs.data[i][j];
            }
        }
    }
}

impl<const M: usize, const N: usize> Sub for Matrix<M, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out -= rhs;
        out
    }
}
impl<const M: usize, const N: usize> SubAssign for Matrix<M, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self.data[i][j] -= rhs.data[i][j];
            }
        }
    }
}

impl<const M: usize, const N: usize> Mul<f64> for Matrix<M, N> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        let mut out = self;
        out *= rhs;
        out
    }
}
impl<const M: usize, const N: usize> MulAssign<f64> for Matrix<M, N> {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..M {
            for j in 0..N {
                self.data[i][j] *= rhs;
            }
        }
    }
}

impl<const M: usize, const N: usize> Mul<Matrix<M, N>> for f64 {
    type Output = Matrix<M, N>;
    fn mul(self, rhs: Matrix<M, N>) -> Self::Output {
        rhs * self
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<Matrix<N, P>> for Matrix<M, N> {
    type Output = Matrix<M, P>;
    fn mul(self, rhs: Matrix<N, P>) -> Self::Output {
        let mut out = Matrix::<M, P>::zeros();
        for i in 0..M {
            for k in 0..N {
                let a = self.data[i][k];
                for j in 0..P {
                    out.data[i][j] += a * rhs.data[k][j];
                }
            }
        }
        out
    }
}
