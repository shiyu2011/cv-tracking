
# ct_conebeam_class.py
# -------------------------------------------------------------
# Cone-beam CT projector in a class:
#   - Intrinsics K (f=SDD in pixels)
#   - Extrinsics [R|t] for given angle (look-at or simple)
#   - Ray-driven forward/backprojection (trilinear sample/splat)
#   - Voxel-driven forward/backprojection (bilinear splat/sample)
# World origin at isocenter (mm). Camera origin at source.
# -------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

class CTProjector:
    def __init__(self,
                 nu: int, nv: int,
                 det_width_mm: float, det_height_mm: float,
                 SID_mm: float, SDD_mm: float,
                 beam_axis: str = "y",
                 use_lookat: bool = True,
                 voxel_size_mm: float = 1.0):
        self.nu = int(nu)
        self.nv = int(nv)
        self.det_w = float(det_width_mm)
        self.det_h = float(det_height_mm)
        self.SID = float(SID_mm)
        self.SDD = float(SDD_mm)
        self.beam_axis = beam_axis
        self.use_lookat = use_lookat
        self.vox = float(voxel_size_mm)

        self.K = self.build_intrinsic()
        self.Kinv = np.linalg.inv(self.K).astype(np.float32)

    # ---------- Geometry builders ----------
    def build_intrinsic(self) -> np.ndarray:
        du = self.det_w / float(self.nu)  # mm/px
        dv = self.det_h / float(self.nv)  # mm/px
        fx = self.SDD / du                 # px
        fy = self.SDD / dv                 # px
        cx, cy = self.nu / 2.0, self.nv / 2.0
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return K

    @staticmethod
    def Rz(angle_rad: float) -> np.ndarray:
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    def build_extrinsic(self, angle_rad: float) -> np.ndarray:
        return (self._build_extrinsic_lookat(angle_rad) if self.use_lookat
                else self._build_extrinsic_simple(angle_rad))

    def _build_extrinsic_simple(self, angle_rad: float) -> np.ndarray:
        # World axes == camera axes at 0°
        if self.beam_axis == "y":
            S0 = np.array([0.0, -self.SID, 0.0], dtype=np.float32)
        elif self.beam_axis == "x":
            S0 = np.array([-self.SID, 0.0, 0.0], dtype=np.float32)
        else:
            raise ValueError("beam_axis must be 'x' or 'y'")
        Cw = self.Rz(angle_rad) @ S0
        R_wc = self.Rz(-angle_rad)
        t = - R_wc @ Cw
        return np.hstack([R_wc, t.reshape(3,1)])

    def _build_extrinsic_lookat(self, angle_rad: float) -> np.ndarray:
        # Robust: rotate S0, D0; build camera axes with look-at
        if self.beam_axis == "y":
            S0 = np.array([0.0, -self.SID, 0.0], dtype=np.float32)
            D0 = np.array([0.0,  self.SDD - self.SID, 0.0], dtype=np.float32)
        elif self.beam_axis == "x":
            S0 = np.array([-self.SID, 0.0, 0.0], dtype=np.float32)
            D0 = np.array([ self.SDD - self.SID, 0.0, 0.0], dtype=np.float32)
        else:
            raise ValueError("beam_axis must be 'x' or 'y'")

        Rw = self.Rz(angle_rad)
        S = Rw @ S0
        D = Rw @ D0

        zc = D - S; zc = zc / np.linalg.norm(zc)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(up, zc))) > 0.999:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        xc = np.cross(up, zc); xc = xc / np.linalg.norm(xc)
        yc = np.cross(zc, xc)

        R_cw = np.stack([xc, yc, zc], axis=1)
        R_wc = R_cw.T
        t = - R_wc @ S
        return np.hstack([R_wc, t.reshape(3,1)])

    @staticmethod
    def camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
        R_wc = extrinsic[:, :3]
        t = extrinsic[:, 3]
        R_cw = R_wc.T
        return - R_cw @ t

    # ---------- Sampling helpers ----------
    @staticmethod
    def _bilinear(img: np.ndarray, u: float, v: float) -> float:
        h, w = img.shape
        if u < 0 or v < 0 or u >= w-1 or v >= h-1:
            return 0.0
        u0, v0 = int(np.floor(u)), int(np.floor(v))
        du, dv = u - u0, v - v0
        a = img[v0,   u0  ]
        b = img[v0,   u0+1]
        c = img[v0+1, u0  ]
        d = img[v0+1, u0+1]
        return float((1-du)*(1-dv)*a + du*(1-dv)*b + (1-du)*dv*c + du*dv*d)

    @staticmethod
    def _splat_bilinear_add(img: np.ndarray, u: float, v: float, add_val: float):
        h, w = img.shape
        u0, v0 = int(np.floor(u)), int(np.floor(v))
        du, dv = u - u0, v - v0

        candidates = [
            (v0,   u0,   (1-du)*(1-dv)),
            (v0,   u0+1, du*(1-dv)),
            (v0+1, u0,   (1-du)*dv),
            (v0+1, u0+1, du*dv),
        ]
        wsum = 0.0
        kept = []
        for iv, iu, wgt in candidates:
            if 0 <= iu < w and 0 <= iv < h and wgt > 0:
                kept.append((iv, iu, wgt))
                wsum += wgt
        if wsum <= 0:
            return
        scale = add_val / wsum
        for iv, iu, wgt in kept:
            img[iv, iu] += wgt * scale

    
    @staticmethod
    def world_to_index(Xw: np.ndarray, voxel_size_mm: float,
                       vol_shape: Tuple[int, int, int]) -> np.ndarray:
        xi = Xw[0] / voxel_size_mm + (vol_shape[2]-1)/2.0
        yi = Xw[1] / voxel_size_mm + (vol_shape[1]-1)/2.0
        zi = Xw[2] / voxel_size_mm + (vol_shape[0]-1)/2.0
        return np.array([xi, yi, zi], dtype=np.float32)


    @staticmethod
    def _sample_trilinear(vol: np.ndarray, Xw: np.ndarray,
                          voxel_size_mm: float,
                          cx: float, cy: float, cz: float) -> float:
        Nz, Ny, Nx = vol.shape
        [xf, yf, zf] = CTProjector.world_to_index(Xw, voxel_size_mm, vol.shape)

        if xf < 0 or xf > Nx-1 or yf < 0 or yf > Ny-1 or zf < 0 or zf > Nz-1:
            return 0.0
        x0 = int(np.floor(xf)); x1 = min(x0+1, Nx-1)
        y0 = int(np.floor(yf)); y1 = min(y0+1, Ny-1)
        z0 = int(np.floor(zf)); z1 = min(z0+1, Nz-1)
        tx = xf - x0; ty = yf - y0; tz = zf - z0
        c000 = vol[z0, y0, x0]; c100 = vol[z0, y0, x1]
        c010 = vol[z0, y1, x0]; c110 = vol[z0, y1, x1]
        c001 = vol[z1, y0, x0]; c101 = vol[z1, y0, x1]
        c011 = vol[z1, y1, x0]; c111 = vol[z1, y1, x1]
        c00 = c000*(1-tx) + c100*tx; c01 = c001*(1-tx) + c101*tx
        c10 = c010*(1-tx) + c110*tx; c11 = c011*(1-tx) + c111*tx
        c0 = c00*(1-ty) + c10*ty;    c1 = c01*(1-ty) + c11*ty
        return float(c0*(1-tz) + c1*tz)

    @staticmethod
    def _splat_trilinear_add(vol: np.ndarray, Xw: np.ndarray, add_val: float,
                             voxel_size_mm: float, cx: float, cy: float, cz: float):
        Nz, Ny, Nx = vol.shape
        [xf, yf, zf] = CTProjector.world_to_index(Xw, voxel_size_mm, vol.shape)

        if xf < -1 or xf > Nx or yf < -1 or yf > Ny or zf < -1 or zf > Nz:
            return
        x0 = int(np.floor(xf)); x1 = x0 + 1
        y0 = int(np.floor(yf)); y1 = y0 + 1
        z0 = int(np.floor(zf)); z1 = z0 + 1
        tx = xf - x0; ty = yf - y0; tz = zf - z0
        w000 = (1-tx)*(1-ty)*(1-tz); w100 = tx*(1-ty)*(1-tz)
        w010 = (1-tx)*ty*(1-tz);     w110 = tx*ty*(1-tz)
        w001 = (1-tx)*(1-ty)*tz;     w101 = tx*(1-ty)*tz
        w011 = (1-tx)*ty*tz;         w111 = tx*ty*tz
        candidates = [
            (z0, y0, x0, w000), (z0, y0, x1, w100),
            (z0, y1, x0, w010), (z0, y1, x1, w110),
            (z1, y0, x0, w001), (z1, y0, x1, w101),
            (z1, y1, x0, w011), (z1, y1, x1, w111),
        ]
        kept = []; wsum = 0.0
        for zi, yi, xi, w in candidates:
            if 0 <= xi < Nx and 0 <= yi < Ny and 0 <= zi < Nz and w > 0:
                kept.append((zi, yi, xi, w)); wsum += w
        if wsum <= 0: return
        scale = add_val / wsum
        for zi, yi, xi, w in kept:
            vol[zi, yi, xi] += w * scale

    # ---------- Projection/backprojection (ray-driven) ----------
    def forward_ray(self, volume: np.ndarray, angle_rad: float,
                    step_mm: Optional[float]=None, s_max_mm: Optional[float]=None) -> np.ndarray:
        Nz, Ny, Nx = volume.shape
        proj = np.zeros((self.nv, self.nu), dtype=np.float32)

        Kinv = self.Kinv
        extr = self.build_extrinsic(angle_rad)
        R_wc = extr[:, :3]; R_cw = R_wc.T
        Cw = self.camera_center_from_extrinsic(extr)

        if step_mm is None:
            step_mm = 0.75 * self.vox

        cx, cy, cz = (Nx-1)/2.0, (Ny-1)/2.0, (Nz-1)/2.0
        hx, hy, hz = Nx*self.vox/2.0, Ny*self.vox/2.0, Nz*self.vox/2.0
        if s_max_mm is None:
            corners = np.array([[ sx,  sy,  sz] for sx in (-hx, hx)
                                           for sy in (-hy, hy)
                                           for sz in (-hz, hz)], dtype=np.float32)
            s_max_mm = float(np.max(np.linalg.norm(corners - Cw, axis=1))) + step_mm

        for iv in range(self.nv):
            v = iv + 0.5
            for iu in range(self.nu):
                u = iu + 0.5
                dc = Kinv @ np.array([u, v, 1.0], dtype=np.float32); dc /= np.linalg.norm(dc)
                dw = R_cw @ dc; dw /= np.linalg.norm(dw)
                s = 0.0; acc = 0.0
                while s <= s_max_mm:
                    Xw = Cw + s * dw #along ray in world coords
                    acc += self._sample_trilinear(volume, Xw, self.vox, cx, cy, cz) #accumulate along ray
                    s += step_mm #accumlate the length
                proj[iv, iu] = acc * step_mm #this is the integral along the ray from exp(-mu*ds)
        return proj
    
    # GPU Cuda accelerated version of forward_ray
    def forward_ray_cuda(self, volume:np.ndarray, angle_rad: float,
                       step_mm: float = None) -> np.ndarray:
        Nz, Ny, Nx = volume.shape
        vox = self.vox
        
        if step_mm is None:
            step_mm = 0.75 * vox
            
        #build extrinsic
        extr = self.build_extrinsic(angle_rad)
        Rwc = extr[:, :3].astype(np.float32) #word to camera rotation 3x3
        t = extr[:, 3].astype(np.float32) #translation 3x1
        
        #Torch tensors on GPU
        vol_t = torch.from_numpy(volume).to(device='cuda', dtype=torch.float32).contiguous()
        img_t = torch.zeros((self.nv, self.nu), device='cuda', dtype=torch.float32).contiguous()
        
        Kinv_t = torch.from_numpy(self.Kinv).to(device='cuda', dtype=torch.float32).contiguous()
        Rwc_t = torch.from_numpy(Rwc).to(device='cuda', dtype=torch.float32).contiguous()
        t_t = torch.from_numpy(t).to(device='cuda', dtype=torch.float32).contiguous()
        
        #origin: voxel index (0 0 0) to world coord mm 
        oxi, oyi, ozi = 0.0, 0.0, 0.0
        ox, oy, oz = self.index_to_world(oxi, oyi, ozi, vox, volume.shape)
        
        fwd_ext.forward_project_intr_extr(
            img_t, vol_t,
            Nx, Ny, Nz,
            vox, vox, vox,
            ox, oy, oz,
            Kinv_t, Rwc_t, t_t,
            step_mm
        )
        return img_t.cpu().numpy()

    def backproject_ray_into(self, vol_shape: Tuple[int,int,int], proj: np.ndarray, angle_rad: float,
                             step_mm: Optional[float]=None, s_max_mm: Optional[float]=None) -> np.ndarray:
        nv, nu = proj.shape
        Nz, Ny, Nx = vol_shape
        vol = np.zeros(vol_shape, dtype=np.float32)

        Kinv = self.Kinv
        extr = self.build_extrinsic(angle_rad)
        R_wc = extr[:, :3]; R_cw = R_wc.T
        Cw = self.camera_center_from_extrinsic(extr)

        if step_mm is None:
            step_mm = 0.75 * self.vox

        cx, cy, cz = (Nx-1)/2.0, (Ny-1)/2.0, (Nz-1)/2.0
        hx, hy, hz = Nx*self.vox/2.0, Ny*self.vox/2.0, Nz*self.vox/2.0
        if s_max_mm is None:
            corners = np.array([[ sx,  sy,  sz] for sx in (-hx, hx)
                                           for sy in (-hy, hy)
                                           for sz in (-hz, hz)], dtype=np.float32)
            s_max_mm = float(np.max(np.linalg.norm(corners - Cw, axis=1))) + step_mm

        for iv in range(self.nv):
            v = iv + 0.5
            for iu in range(self.nu):
                val = proj[iv, iu]
                if val == 0.0: continue
                u = iu + 0.5
                dc = Kinv @ np.array([u, v, 1.0], dtype=np.float32); dc /= np.linalg.norm(dc)
                dw = R_cw @ dc; dw /= np.linalg.norm(dw)

                # pass 1: count inside
                s = 0.0; count = 0
                while s <= s_max_mm:
                    Xw = Cw + s * dw
                    xi = int(round(Xw[0]/self.vox + cx))
                    yi = int(round(Xw[1]/self.vox + cy))
                    zi = int(round(Xw[2]/self.vox + cz))
                    if 0 <= xi < Nx and 0 <= yi < Ny and 0 <= zi < Nz:
                        count += 1
                    s += step_mm
                if count == 0: continue
                w = val / count #along ray weight, to make sure integral is correct


                # pass 2: trilinear splat
                s = 0.0
                while s <= s_max_mm:
                    Xw = Cw + s * dw
                    self._splat_trilinear_add(vol, Xw, w, self.vox, cx, cy, cz)
                    s += step_mm

        return vol
    
    @staticmethod
    def index_to_world(xi: int, yi: int, zi: int, voxel_size_mm: float,
                       vol_shape: Tuple[int, int, int] ) -> np.ndarray:
        xw = (xi - (vol_shape[2]-1)/2.0) * voxel_size_mm 
        yw = (yi - (vol_shape[1]-1)/2.0) * voxel_size_mm
        zw = (zi - (vol_shape[0]-1)/2.0) * voxel_size_mm
        return np.array([xw, yw, zw], dtype=np.float32)


    # ---------- Voxel-driven (scatter/gather) ----------
    def forward_voxel(self, volume: np.ndarray, angle_rad: float) -> np.ndarray:
        Nz, Ny, Nx = volume.shape
        proj = np.zeros((self.nv, self.nu), dtype=np.float32)
        extr = self.build_extrinsic(angle_rad)
        P = self.K @ extr

        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    val = volume[z, y, x]
                    if val == 0.0: continue
                    [wx, wy, wz] = self.index_to_world(x, y, z, self.vox, volume.shape)
                    Xh = np.array([wx, wy, wz, 1.0], dtype=np.float32)
                    p = P @ Xh
                    u = p[0]/p[2]; v = p[1]/p[2]
                    self._splat_bilinear_add(proj, u, v, val)
        return proj

    def backproject_voxel(self, proj: np.ndarray, angle_rad: float,
                          vol_shape: Tuple[int,int,int]) -> np.ndarray:
        nv, nu = proj.shape
        Nz, Ny, Nx = vol_shape
        vol = np.zeros(vol_shape, dtype=np.float32)
        extr = self.build_extrinsic(angle_rad)

        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    [wx, wy, wz] = self.index_to_world(x, y, z, self.vox, vol_shape)
                    Xh = np.array([wx, wy, wz, 1.0], dtype=np.float32)
                    p = self.K @ (extr @ Xh)
                    u = p[0]/p[2]; v = p[1]/p[2]
                    vol[z, y, x] += self._bilinear(proj, u, v)
        return vol

    
    @staticmethod
    def shepp_logan_3d(nx: int, ny: int, nz: int) -> np.ndarray:
        phantom = shepp_logan_phantom()
        phantom_3d = np.stack([phantom]*nz, axis=0)
        phantom_3d = resize(phantom_3d, (nz, ny, nx), order=1, mode='constant', cval=0.0, anti_aliasing=True)
        return phantom_3d.astype(np.float32)
    
    


import torch
from torch.utils.cpp_extension import load
#Compile Cuda Extension (only once, cached afterwards)
fwd_ext = load(
    name="fwd_intr_extr_ext",
    sources=["forward_intr_extr.cu"],
    extra_cuda_cflags=['--use_fast_math', "-O3"],
    verbose=True
)
# ------------------- One-angle quick test -------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Small phantom (cube) to keep runtime fast

    Nx = Ny = 128
    Nz = 32
    nu = nv = 96
    vox = 1.0
    SID, SDD = 500.0, 1000.0
    det_w = det_h = 360.0  # mm
    angle_deg = 20.0
    angle = np.deg2rad(angle_deg)

    # Build phantom

    # Projector
    ct = CTProjector(nu, nv, det_w, det_h, SID, SDD, beam_axis="y",
                     use_lookat=True, voxel_size_mm=vox)
    
    vol = ct.shepp_logan_3d(nx=Nx, ny=Ny, nz=Nz)

    # Ray-driven forward/back
    # profile the run time of forward_ray and fword_ray_cuda
    
    import time
    start = time.time()
    proj_ray = ct.forward_ray(vol, angle, step_mm=0.75*vox)
    end = time.time()
    cpuT = end - start
    print("forward_ray time in cpu: ", cpuT)
    
    torch.cuda.synchronize()
    start = time.time()
    proj_ray_cuda = ct.forward_ray_cuda(vol, angle, step_mm=0.75*vox)
    end = time.time()
    torch.cuda.synchronize()
    gpuT = end - start
    print("fword_ray_cuda time in gpu: ", gpuT)
    
    
    
    recon_ray = ct.backproject_ray_into(vol.shape, proj_ray, angle, step_mm=0.75*vox)

    # Voxel-driven forward/back
    proj_vox = ct.forward_voxel(vol, angle)
    recon_vox = ct.backproject_voxel(proj_vox, angle, vol.shape)

    print("Sum proj (ray):   ", float(np.sum(proj_ray)))
    print("Sum proj (voxel): ", float(np.sum(proj_vox)))
    print("Sum recon (ray):  ", float(np.sum(recon_ray)))
    print("Sum recon (voxel):", float(np.sum(recon_vox)))

    # Visualize middle z-slice
    midz = Nz//2
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    axs[0,0].imshow(vol[midz], cmap="gray"); axs[0,0].set_title("Phantom z-slice")
    axs[0,1].imshow(proj_ray, cmap="gray");       axs[0,1].set_title(f"Proj (ray)  {cpuT:.4f}s")
    axs[0,2].imshow(proj_ray_cuda, cmap="gray");  axs[0,2].set_title(f"Proj (ray + CUDA) {gpuT:.4f}s")
    axs[0,3].imshow(recon_ray[midz], cmap="gray");axs[0,3].set_title("Backproj (ray)")

    
    axs[1,0].imshow(vol[midz], cmap="gray"); axs[1,0].set_title("Phantom (repeat)")
    axs[1,1].imshow(proj_vox, cmap="gray");  axs[1,1].set_title("Proj (voxel)")
    axs[1,2].imshow(proj_vox, cmap="gray");  axs[1,2].set_title("Proj (voxel) (repeat)")
    axs[1,3].imshow(recon_vox[midz], cmap="gray"); axs[1,3].set_title("Backproj (voxel)")
    for ax in axs.ravel(): ax.axis("off")
    plt.tight_layout()

    plt.savefig("ct_conebeam_test.png", dpi=150)
    plt.close()
