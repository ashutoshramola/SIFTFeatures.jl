#!/usr/bin/env python3
"""OpenCV integration checks. No cv2/Python dependency enters the Julia package."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from common import execute


def fixtures():
    rng = np.random.default_rng(314159)
    a = np.full((256, 320), 32, np.uint8)
    for _ in range(170):
        x, y = int(rng.integers(12, 308)), int(rng.integers(12, 244))
        r = int(rng.integers(2, 12))
        value = int(rng.integers(50, 240))
        if rng.random() < 0.5:
            cv2.circle(a, (x, y), r, value, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(a, (x-r, y-r), (x+r, y+r), value, -1)
    cv2.putText(a, "SIFT 128", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2, cv2.LINE_AA)
    a = cv2.GaussianBlur(a, (3, 3), 0.5)
    h, w = a.shape
    transforms = {
        "translate": np.array([[1., 0, 9], [0, 1, 7], [0, 0, 1]]),
        "rotate90": np.array([[0., 1, 0], [-1, 0, w-1], [0, 0, 1]]),
        "similarity": np.vstack([cv2.getRotationMatrix2D((w/2, h/2), 23, 1.15), [0, 0, 1]]),
        "perspective": np.array([[1.02, 0.08, -9], [-0.035, 0.96, 11], [0.00025, -0.0002, 1]]),
    }
    images = {"base": a}
    for name, H in transforms.items():
        images[name] = np.rot90(a).copy() if name == "rotate90" else cv2.warpPerspective(a, H, (w,h))
    # Independent continuous texture exercises a different distribution of extrema.
    noise = rng.uniform(0, 255, (255, 319)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 1.0)
    texture = np.clip((texture-128)*3+128, 0, 255).astype(np.uint8)
    images["texture"] = texture
    return images, transforms


def normalized(d):
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-15)


def angles(a, b):
    return np.abs((a-b+np.pi) % (2*np.pi)-np.pi)


def summarize(v):
    if len(v) == 0:
        return None
    return {"min": float(np.min(v)), "p05":float(np.percentile(v,5)), "median": float(np.median(v)), "p95": float(np.percentile(v,95)), "max": float(np.max(v))}


def frame_pairs(j, ref):
    # One-to-one geometry-only correspondence; descriptor similarity cannot select
    # the pairs used to score descriptor similarity. Multiple orientations remain distinct.
    # A one-pixel spatial grid avoids N×M matrices on textured photographs.
    # Candidate bounds, costs, and tie order equal the dense implementation.
    grid = {}
    for r, p in enumerate(ref):
        grid.setdefault(tuple(np.floor(p[:2]).astype(int)), []).append(r)
    candidates = []
    for i, p in enumerate(j):
        x, y = np.floor(p[:2]).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for r in grid.get((x+dx, y+dy), ()):
                    pos = np.linalg.norm(p[:2]-ref[r,:2])
                    scale = abs(np.log(p[2]/ref[r,2]))
                    angle = float(angles(p[3], ref[r,3]))
                    if pos < 1 and scale < 0.05 and angle < np.deg2rad(5):
                        candidates.append((pos+10*scale+angle, i, r))
    order = [(i,r) for _,i,r in sorted(candidates)]
    used_j,used_r,pairs = set(),set(),[]
    for i,r in order:
        if i not in used_j and r not in used_r:
            pairs.append((i,r)); used_j.add(i); used_r.add(r)
    return np.asarray(pairs,dtype=int).reshape(-1,2)


def ratio_matches(a,b,crosscheck=True):
    if len(a)==0 or len(b)<2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = [(m.queryIdx,m.trainIdx) for m,n in bf.knnMatch(a,b,k=2) if m.distance<0.8*n.distance]
    if crosscheck:
        reverse = {m.queryIdx:m.trainIdx for m in bf.match(b,a)}
        pairs = [(i,j) for i,j in pairs if reverse[j]==i]
    return pairs


def geometric_errors(pairs, a, b, H):
    pts = np.c_[a[:,:2],np.ones(len(a))] @ H.T
    pts = pts[:,:2]/pts[:,2:]
    return np.array([np.linalg.norm(pts[i]-b[j,:2]) for i,j in pairs])


def run(args, report, gates):
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.04,
                         edgeThreshold=10, sigma=1.6, enable_precise_upscale=True)
    images, transforms = fixtures()
    report.update({"opencv":cv2.__version__,"numpy":np.__version__,
              "julia":subprocess.check_output([args.julia,"--startup-file=no","--version"],text=True).strip(),
              "precise_upscale":True,"images":{},"transforms":{}})
    with tempfile.TemporaryDirectory(prefix="sift-validation-") as tmp:
        work = Path(tmp)
        probes = np.array([[100.5,104.5,2.5,np.pi/4,0,2],
                           [141.,141.,4.5,np.pi/2,1,1],
                           [80.25,82.25,1.1,0.0,-1,1],
                           [180.5,170.5,3.0,2*np.pi-1e-5,0,3]])
        np.savetxt(work/"probes.csv",probes,delimiter=',')
        (work/"images.txt").write_text('\n'.join(images))
        for name,img in images.items():
            img.tofile(work/(name+".u8"))
            (work/(name+".shape")).write_text(f"{img.shape[0]} {img.shape[1]}")
        subprocess.run([args.julia,"--startup-file=no",f"--project={ROOT}",str(ROOT/"validation/export.jl"),tmp],check=True)
        native,reference = {},{}
        for name,img in images.items():
            j = np.loadtxt(work/(name+".kp.csv"),delimiter=',',skiprows=1,ndmin=2)
            jd = np.fromfile(work/(name+".desc.f32"),dtype=np.float32).reshape(-1,128)
            rk,rd = sift.detectAndCompute(img,None)
            ref = np.array([[*p.pt,p.size/2,np.deg2rad(p.angle)] for p in rk])
            rd = normalized(rd)
            native[name],reference[name] = (j,jd),(ref,rd)
            pairs = frame_pairs(j,ref)
            if not gates.require(len(pairs)>30, name+": corresponding frames", len(pairs)):
                continue
            i,r = pairs.T
            fixed = [cv2.KeyPoint(float(p[0]),float(p[1]),float(2*p[2]),
                                  float(np.rad2deg(p[3])%360),float(p[4]),
                                  (int(p[5])&255)|(int(p[6])<<8)) for p in j]
            _,fd = sift.compute(img,fixed)
            fixed_cos = np.sum(jd*normalized(fd),axis=1)
            cos = np.sum(jd[i]*rd[r],axis=1)
            probe_keys = [cv2.KeyPoint(float(p[0]),float(p[1]),float(2*p[2]),
                                      float(np.rad2deg(p[3])%360),1.,
                                      (int(p[4])&255)|(int(p[5])<<8)) for p in probes]
            _,probe_ref = sift.compute(img,probe_keys)
            probe_native = np.fromfile(work/(name+".probes.f32"),dtype=np.float32).reshape(-1,128)
            probe_cos = np.sum(probe_native*normalized(probe_ref),axis=1)
            gates.require(float(np.min(probe_cos)) > 0.995, name+": half-pixel descriptors", float(np.min(probe_cos)))
            stats = {
                "half_pixel_probe_cosine": summarize(probe_cos),
                "native_count":len(j),"opencv_count":len(ref),"paired_count":len(pairs),
                "native_coverage":len(pairs)/len(j),"opencv_coverage":len(pairs)/len(ref),
                "position_px":summarize(np.linalg.norm(j[i,:2]-ref[r,:2],axis=1)),
                "relative_scale_error":summarize(np.abs(j[i,2]/ref[r,2]-1)),
                "angle_error_deg":summarize(np.rad2deg(angles(j[i,3],ref[r,3]))),
                "detected_descriptor_cosine":summarize(cos),
                "fixed_frame_descriptor_cosine":summarize(fixed_cos),
            }
            report["images"][name] = stats
            print(name,json.dumps(stats),flush=True)
            # Gates are intentionally toleranced for floating point and orientation bin boundaries.
            gates.require(min(stats["native_coverage"],stats["opencv_coverage"]) >= 0.97, name+": frame coverage", min(stats["native_coverage"],stats["opencv_coverage"]))
            gates.require(stats["position_px"]["p95"] < 0.01, name+": position p95", stats["position_px"]["p95"])
            gates.require(stats["relative_scale_error"]["p95"] < 0.005, name+": scale p95", stats["relative_scale_error"]["p95"])
            gates.require(stats["angle_error_deg"]["p95"] < 0.1, name+": angle p95", stats["angle_error_deg"]["p95"])
            gates.require(float(np.median(cos)) > 0.999, name+": descriptor median cosine", float(np.median(cos)))
            gates.require(float(np.percentile(fixed_cos,5)) > 0.995, name+": fixed-frame descriptor p05", float(np.percentile(fixed_cos,5)))
            # Cross-implementation descriptors must retrieve the corresponding frame.
            cross = ratio_matches(jd,rd)
            cross_ok = sum(np.linalg.norm(j[q,:2]-ref[t,:2])<1 and
                           abs(np.log(j[q,2]/ref[t,2]))<0.05 and
                           angles(j[q,3],ref[t,3])<np.deg2rad(5) for q,t in cross)
            stats["cross_matches"] = len(cross)
            stats["cross_match_precision"] = cross_ok/max(1,len(cross))
            gates.require(len(cross) >= 0.8*min(len(j),len(ref)), name+": cross-library retrieval count", len(cross))
            gates.require(stats["cross_match_precision"] >= 0.95, name+": cross-library precision", stats["cross_match_precision"])
        for name,H in transforms.items():
            a,ad = native["base"]; b,bd = native[name]
            matches = np.loadtxt(work/(name+".matches.csv"),delimiter=',',skiprows=1,ndmin=2)
            pairs = [tuple(map(int,p[:2])) for p in matches]
            gates.require(set(pairs)==set(ratio_matches(ad,bd)), name+": same-descriptor matcher agreement")
            errors = geometric_errors(pairs,a,b,H)
            ra,rad = reference["base"]; rb,rbd = reference[name]
            rpairs = ratio_matches(rad,rbd)
            rerrors = geometric_errors(rpairs,ra,rb,H)
            stats = {"native_matches":len(pairs),"native_inlier_fraction":float(np.mean(errors<2)) if len(errors) else 0.,
                     "native_error_px":summarize(errors),"opencv_matches":len(rpairs),
                     "opencv_inlier_fraction":float(np.mean(rerrors<2)) if len(rerrors) else 0.}
            report["transforms"][name] = stats
            print(name,json.dumps(stats),flush=True)
            gates.require(len(pairs)>=40, name+": match count", len(pairs))
            gates.require(stats["native_inlier_fraction"] >= 0.9, name+": inlier fraction", stats["native_inlier_fraction"])
            # Recover a homography from descriptor matches; compare four interior probes.
            if not gates.require(len(pairs)>=4, name+": sufficient matches for homography", len(pairs)):
                continue
            pa = np.float32([a[i,:2] for i,j in pairs]); pb = np.float32([b[j,:2] for i,j in pairs])
            cv2.setRNGSeed(7)
            fit,_ = cv2.findHomography(pa,pb,cv2.RANSAC,2)
            if not gates.require(fit is not None, name+": homography found"):
                continue
            probes = np.float64([[60,60,1],[260,60,1],[60,200,1],[260,200,1]])
            def project(M):
                p = probes @ M.T
                return p[:,:2]/p[:,2:]
            error = float(np.max(np.linalg.norm(project(fit)-project(H),axis=1)))
            stats["recovered_homography_max_probe_error_px"] = error
            gates.require(error<1.5, name+": homography probe error", error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia",default=os.environ.get("JULIA","julia"))
    parser.add_argument("--output",default=str(ROOT/"validation/latest-results.json"))
    sys.exit(execute(parser.parse_args(), run, "synthetic OpenCV reference"))
