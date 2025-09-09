# Homogeneous Geometry — One‑Page Quick Reference (GitHub‑friendly)

This sheet collects the identities and formulas you’ll use for two‑view geometry, stereo depth, and epipolar constraints. Keep it next to your code.

---

## 1) Homogeneous coordinates (why we use them)
- Make **points linear**: camera projection and rigid transforms become matrix multiplies.
- Represent **points at infinity** with $w=0$ (vanishing points, parallel lines).
- Compose transforms by **matrix multiplication** (translations included).

**2D point (image):** $\tilde{x} = [u,\ v,\ 1]^T$ (any nonzero scale is same point).  
**2D line:** $\ell = [a,\ b,\ c]^T$ represents $a\,u + b\,v + c = 0$.  
**3D point:** $X_h = [X,\ Y,\ Z,\ 1]^T$.  
**3D plane:** $\pi = [\pi_1,\pi_2,\pi_3,\pi_4]^T$ with $\pi^T X_h = 0$.

**Incidence:** $\ell^T \tilde{x} = 0$.  
**Line through two points:** $\ell = \tilde{x}_1 \times \tilde{x}_2$.  
**Intersection of two lines:** $\tilde{x} = \ell_1 \times \ell_2$.

**Dehomogenize:** $(u,v) = (\tilde{u}/\tilde{w},\ \tilde{v}/\tilde{w})$.  
**At infinity:** $\tilde{w}=0$.

---

## 2) Camera model
$$
\tilde{x} \sim P\,X_h,\quad P = K\,[R\ |\ t],\quad
K=\begin{bmatrix} f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}
$$

- **Pixels:** $\tilde{x} = [\tilde{u},\tilde{v},\tilde{w}]^T \Rightarrow (u,v)=(\tilde{u}/\tilde{w},\,\tilde{v}/\tilde{w})$.  
- **Normalized rays:** $x = K^{-1}\tilde{x} = [X/Z,\ Y/Z,\ 1]^T$.  
- **Projective depth:** $\tilde{w} = Z$ (for standard $K$); enforce **cheirality:** $Z>0$.

---

## 3) Epipolar geometry
**Essential:** $x_2^T\,E\,x_1 = 0,\quad E = [t]_\times R$ (normalized coords).  
**Fundamental:** $\tilde{x}_2^T\,F\,\tilde{x}_1 = 0,\quad E = K^T F K$ (pixel coords).

**Epipolar line in image 2 from $\tilde{x}_1$:** $\ell_2 = F\,\tilde{x}_1$.  
**Dual line in image 1 from $\tilde{x}_2$:** $\ell_1 = F^T\,\tilde{x}_2$.

**Sampson distance (robust residual for RANSAC):**
$$
d_S(x_1,x_2;F)=\frac{(\tilde{x}_2^T F \tilde{x}_1)^2}
{(F\tilde{x}_1)_1^2 + (F\tilde{x}_1)_2^2 + (F^T\tilde{x}_2)_1^2 + (F^T\tilde{x}_2)_2^2}
$$

---

## 4) Triangulation (DLT, two arbitrary views)
Given $P_1, P_2$ and matches $(u_1,v_1)$, $(u_2,v_2)$, build the $4\times4$ system:
$$
A=\begin{bmatrix}
u_1 P_{1,3}-P_{1,1}\\
v_1 P_{1,3}-P_{1,2}\\
u_2 P_{2,3}-P_{2,1}\\
v_2 P_{2,3}-P_{2,2}
\end{bmatrix},\quad A\,X_h=0.
$$
Solve by SVD (right singular vector for smallest singular value), then normalize $X_h$.  
**Cheirality test:** $Z_1>0$ and $Z_2>0$ (point in front of both cameras).

**Reprojection error (pixels):** for $i\in\{1,2\}$
$$
\tilde{x}_i=P_iX_h,\quad \hat{u}_i=\tilde{x}_{i,1}/\tilde{x}_{i,3},\ \hat{v}_i=\tilde{x}_{i,2}/\tilde{x}_{i,3},\quad
e_i=\sqrt{(\hat{u}_i-u_i)^2+(\hat{v}_i-v_i)^2}.
$$

---

## 5) Rectified stereo (fast closed form)
For left/right rectified pair with baseline $B$ (right shifted +X from left, so $t=[-B,0,0]^T$):  
**Disparity:** $d=u_L-u_R\ (>0)$.  
**Depth:** $ Z = \dfrac{f_x\,B}{d} $.  
**Back‑project:** $ X=\dfrac{(u-c_x)Z}{f_x},\quad Y=\dfrac{(v-c_y)Z}{f_y} $.

**Practical gates:** require $d>0$ and not too small (e.g. $d>10^{-3}$) to avoid $Z\to\infty$.

---

## 6) Rigid transforms in homogeneous form
**2D:** $ \tilde{x}' = \begin{bmatrix}R& t\\ 0&1\end{bmatrix}\tilde{x} $ (3×3).  
**3D:** $ X'_h = \begin{bmatrix}R& t\\ 0&1\end{bmatrix} X_h $ (4×4).  
Compose by multiplying matrices; translations are included naturally.

---

## 7) Common pitfalls (checklist)
- **Undistort** pixels before geometry unless $P$ models distortion.  
- Don’t mix **pixels** ($\tilde{x}$) with **normalized** ($x$); use $K$ consistently.  
- **Baseline units:** meters; **focal** $f_x,f_y$: pixels.  
- **Sign of $t$/baseline:** for rectified left→right, $t=[-B,0,0]^T$ in the right camera’s $P$.  
- Ensure **disparity $>0$** and **positive depth** in each view.  
- Use **Sampson error** (or epipolar distance) when RANSACing $F$.

---

## 8) Tiny code fragments
**Dehomogenize safely (NumPy broadcasting):**
```python
proj = (P @ X_h.T).T        # (N,3)
uv   = proj[:, :2] / proj[:, 2:3]
```

**Epipolar lines and inlier mask:**
```python
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
l2 = (F @ np.c_[pts1, np.ones(len(pts1))].T).T  # lines in image 2
```
---

**Rule of thumb:** write everything in homogeneous form, keep it linear as long as possible, and only divide by the last coordinate at the very end.
