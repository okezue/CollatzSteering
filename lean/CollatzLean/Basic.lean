import Mathlib.Tactic

set_option linter.style.nativeDecide false
set_option linter.style.longLine false

noncomputable section
open Nat

def v2 : ℕ → ℕ
  | 0 => 0
  | n + 1 => if (n + 1) % 2 = 0 then 1 + v2 ((n + 1) / 2) else 0

def kCol (n : ℕ) : ℕ := v2 (n + 1)
def mCol (n : ℕ) : ℕ := (n + 1) / 2 ^ kCol n
def apx (n : ℕ) : ℕ := 3 ^ kCol n * mCol n - 1
def kpCol (n : ℕ) : ℕ := v2 (apx n)
def kap (n : ℕ) : ℕ := apx n / 2 ^ kpCol n

theorem v2_odd (n : ℕ) (h : n % 2 = 1) : v2 n = 0 := by
  cases n with
  | zero => simp at h
  | succ m => simp [v2, h]

theorem v2_even (n : ℕ) (h : 0 < n) (he : n % 2 = 0) :
    v2 n = 1 + v2 (n / 2) := by
  cases n with
  | zero => omega
  | succ m => simp [v2, he]

theorem v2_two : v2 2 = 1 := by native_decide
theorem v2_four : v2 4 = 2 := by native_decide
theorem v2_eight : v2 8 = 3 := by native_decide
theorem kCol_one : kCol 1 = 1 := by native_decide
theorem kCol_three : kCol 3 = 2 := by native_decide
theorem kCol_seven : kCol 7 = 3 := by native_decide
theorem kap_one : kap 1 = 1 := by native_decide
theorem kap_three : kap 3 = 1 := by native_decide
theorem kap_seven : kap 7 = 13 := by native_decide
theorem kap_nine : kap 9 = 7 := by native_decide
theorem kap_thirteen : kap 13 = 5 := by native_decide
theorem kap_fifteen : kap 15 = 5 := by native_decide

end

section ErrorRatio
open Nat

theorem wrong_kp_ratio (n k kp kpw : ℕ) (hkpw : kpw ≤ kp)
    (a : ℕ) (ha : a = 3 ^ k * ((n + 1) / 2 ^ k) - 1)
    (hdiv : 2 ^ kp ∣ a) :
    a / 2 ^ kpw = (a / 2 ^ kp) * 2 ^ (kp - kpw) := by
  rcases hdiv with ⟨c, hc⟩
  subst hc
  rw [Nat.mul_div_cancel_left _ (by positivity : 0 < 2 ^ kp)]
  rw [show 2 ^ kp * c / 2 ^ kpw = c * 2 ^ (kp - kpw) from by
    rw [mul_comm]
    rw [Nat.mul_div_assoc c (pow_dvd_pow 2 hkpw)]
    congr 1
    exact Nat.pow_div hkpw (by omega)]

theorem wrong_kp_ratio_mul (a kp kpw : ℕ) (hkpw : kpw ≤ kp)
    (hdiv : 2 ^ kp ∣ a) :
    a / 2 ^ kpw = (a / 2 ^ kp) * 2 ^ (kp - kpw) := by
  rcases hdiv with ⟨c, hc⟩
  subst hc
  rw [Nat.mul_div_cancel_left _ (by positivity : 0 < 2 ^ kp)]
  rw [show 2 ^ kp * c / 2 ^ kpw = c * 2 ^ (kp - kpw) from by
    rw [mul_comm]
    rw [Nat.mul_div_assoc c (pow_dvd_pow 2 hkpw)]
    congr 1
    exact Nat.pow_div hkpw (by omega)]

end ErrorRatio

section TwoFormula

def kapWith (n kw kpw : ℕ) : ℕ :=
  let m := (n + 1) / 2 ^ kw
  let a := 3 ^ kw * m - 1
  a / 2 ^ kpw

theorem kapWith_11_eq (n : ℕ) (hn : n % 2 = 1) :
    kapWith n 1 1 = (3 * ((n + 1) / 2) - 1) / 2 := by
  simp [kapWith]

theorem kapWith_21_eq (n : ℕ) :
    kapWith n 2 1 = (9 * ((n + 1) / 4) - 1) / 2 := by
  simp [kapWith]

theorem wrong_kp_gives_ratio (a kp d : ℕ) (hd : d ≤ kp)
    (hdiv : 2 ^ kp ∣ a) :
    a / 2 ^ (kp - d) = (a / 2 ^ kp) * 2 ^ d := by
  have h : kp - d ≤ kp := Nat.sub_le kp d
  rw [wrong_kp_ratio_mul a kp (kp - d) h hdiv]
  congr 1
  congr 1
  exact Nat.sub_sub_self hd

end TwoFormula

section Suffix

theorem v2_dvd : ∀ (n : ℕ), 0 < n → 2 ^ v2 n ∣ n := by
  intro n
  cases n with
  | zero => omega
  | succ m =>
    intro _
    unfold v2
    split_ifs with h
    · have hd : 2 ∣ (m + 1) := Nat.dvd_of_mod_eq_zero h
      have hm : 0 < (m + 1) / 2 := Nat.div_pos (by omega) (by omega)
      have ih := v2_dvd _ hm
      rw [show 1 + v2 ((m + 1) / 2) = v2 ((m + 1) / 2) + 1 from by omega]
      rw [pow_succ]
      rcases ih with ⟨c, hc⟩
      refine ⟨c, ?_⟩
      have h1 := (Nat.div_mul_cancel hd).symm
      rw [hc] at h1
      linarith
    · simp

theorem v2_ge_of_dvd (n : ℕ) (hn : 0 < n) (k : ℕ) (hdvd : 2 ^ k ∣ n) :
    v2 n ≥ k := by
  induction k generalizing n with
  | zero => omega
  | succ k ih =>
    cases n with
    | zero => omega
    | succ m =>
      rw [pow_succ, mul_comm] at hdvd
      have h2 : 2 ∣ (m + 1) := dvd_trans ⟨2^k, rfl⟩ hdvd
      have heven : (m + 1) % 2 = 0 := Nat.mod_eq_zero_of_dvd h2
      rw [v2_even (m + 1) hn heven]
      have hd2 : 2 ^ k ∣ (m + 1) / 2 := by
        rwa [Nat.dvd_div_iff_mul_dvd h2]
      have hpos : 0 < (m + 1) / 2 := Nat.div_pos (by omega) (by omega)
      have := ih ((m + 1) / 2) hpos hd2
      omega

theorem kCol_ge_of_dvd (n k : ℕ) (hdvd : 2 ^ k ∣ (n + 1)) :
    kCol n ≥ k := by
  unfold kCol
  exact v2_ge_of_dvd (n + 1) (by omega) k hdvd

theorem kCol_from_dvd (n k : ℕ) (hdvd : 2 ^ k ∣ (n + 1))
    (hodd : ¬ 2 ^ (k + 1) ∣ (n + 1)) :
    kCol n = k := by
  have hge := kCol_ge_of_dvd n k hdvd
  by_contra hne
  have hgt : kCol n ≥ k + 1 := by omega
  have hv := v2_dvd (n + 1) (by omega)
  unfold kCol at hgt
  exact hodd (dvd_trans (pow_dvd_pow 2 hgt) hv)

end Suffix

section Distribution

open Finset in
theorem count_dvd_shift (N d : ℕ) (hd : 0 < d) :
    (filter (fun i => d ∣ (i + 1)) (range N)).card ≤ N / d + 1 := by
  calc (filter (fun i => d ∣ (i + 1)) (range N)).card
      ≤ (range (N / d + 1)).card := by
        apply Finset.card_le_card_of_injOn (fun i => (i + 1) / d - 1)
        · intro x hx
          simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range] at hx ⊢
          have hle : d ≤ x + 1 := Nat.le_of_dvd (by omega) hx.2
          have h1 : 1 ≤ (x + 1) / d := Nat.div_pos hle hd
          have h2 : (x + 1) / d ≤ N / d := Nat.div_le_div_right (by omega)
          omega
        · intro x₁ hx₁ x₂ hx₂ heq
          simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range] at hx₁ hx₂
          have e1 : (x₁ + 1) / d * d = x₁ + 1 := Nat.div_mul_cancel hx₁.2
          have e2 : (x₂ + 1) / d * d = x₂ + 1 := Nat.div_mul_cancel hx₂.2
          have hle1 : d ≤ x₁ + 1 := Nat.le_of_dvd (by omega) hx₁.2
          have hle2 : d ≤ x₂ + 1 := Nat.le_of_dvd (by omega) hx₂.2
          have h1 : 1 ≤ (x₁ + 1) / d := Nat.div_pos hle1 hd
          have h2 : 1 ≤ (x₂ + 1) / d := Nat.div_pos hle2 hd
          have heq' : (x₁ + 1) / d = (x₂ + 1) / d := by
            have := heq; simp only at this; omega
          linarith [e1.symm.trans (congrArg (· * d) heq' ▸ e2)]
      _ = N / d + 1 := Finset.card_range _

theorem count_odd_with_v2_ge (N j : ℕ) (hj : 0 < j) :
    (Finset.filter (fun i => 2 ^ j ∣ (2 * i + 2))
      (Finset.range N)).card ≤ N / 2 ^ (j - 1) + 1 := by
  have hj1 : j = (j - 1) + 1 := by omega
  have heq : ∀ i, (2 ^ j ∣ (2 * i + 2)) ↔ (2 ^ (j - 1) ∣ (i + 1)) := by
    intro i
    rw [show 2 * i + 2 = 2 * (i + 1) from by ring]
    rw [hj1, pow_succ, mul_comm]
    exact Nat.mul_dvd_mul_iff_left (by omega : 0 < 2)
  simp_rw [heq]
  exact count_dvd_shift N (2 ^ (j - 1)) (by positivity)

end Distribution
