/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void Init3dGradsCUDA(
	const int P, int D, int M,
	float* dF_dopacity,
	float* dF_dshs,
	float* dF_dpos,
	float* dF_drot,
	float* dF_dscale,
	float* dF_dcov3D
	)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P*D*M)
		return;
	
	dF_dshs[idx] = 0.0;

	if (idx < P){
		dF_dopacity[idx] = 0.0;
	}
	if (idx < P*3){
		dF_dpos[idx] = 0.0;
		dF_dscale[idx] = 0.0;
	}
	if (idx < P*4){
		dF_drot[idx] = 0.0;
	}
	if (idx < P*9){
		dF_dcov3D[idx] = 0.0;
	}

}


__global__ void computeGradsForAdaCov3D(
	int P,
	float* dF_dcov3D,
	const float* ada_lpf_ratio
)
{
	int gs_idx = cg::this_grid().thread_rank();
	if (gs_idx >= P)
		return;
	
	// dF_dcov3D[9*gs_idx + 0] += ada_lpf_ratio[3*gs_idx + 0];
	// dF_dcov3D[9*gs_idx + 4] += ada_lpf_ratio[3*gs_idx + 1];
	// dF_dcov3D[9*gs_idx + 8] += ada_lpf_ratio[3*gs_idx + 2];
	return;
}

__global__ void computeGradsFromCov3D(
	int P, 
	const float* rot_cuda,
	const float* scale_cuda,
	const float* dF_dcov3D,
	float* dF_drot,
	float* dF_dscale,
	float* ada_lpf_ratio,
	bool* opt_options_cuda
	)
{
	int gs_idx = cg::this_grid().thread_rank();
	if (gs_idx >= P)
		return;

	glm::mat3 S = glm::mat3(1.0f);

	// if (opt_options_cuda[9]){
	// 	S[0][0] = exp(scale_cuda[gs_idx*3 + 0])*(1.0f + ada_lpf_ratio[gs_idx*3 + 0]);
	// 	S[1][1] = exp(scale_cuda[gs_idx*3 + 1])*(1.0f + ada_lpf_ratio[gs_idx*3 + 1]);
	// 	S[2][2] = exp(scale_cuda[gs_idx*3 + 2])*(1.0f + ada_lpf_ratio[gs_idx*3 + 2]);
	// }
	// else{
	// 	S[0][0] = exp(scale_cuda[gs_idx*3 + 0])+1e-3f;
	// 	S[1][1] = exp(scale_cuda[gs_idx*3 + 1])+1e-3f;
	// 	S[2][2] = exp(scale_cuda[gs_idx*3 + 2])+1e-3f;
	// }

	S[0][0] = exp(scale_cuda[gs_idx*3 + 0]);
	S[1][1] = exp(scale_cuda[gs_idx*3 + 1]);
	S[2][2] = exp(scale_cuda[gs_idx*3 + 2]);

	float r = rot_cuda[gs_idx*4 + 0];
	float x = rot_cuda[gs_idx*4 + 1];
	float y = rot_cuda[gs_idx*4 + 2];
	float z = rot_cuda[gs_idx*4 + 3];
	float norm = sqrt(r*r + x*x + y*y + z*z);
	r = r / norm;
	x = x / norm;
	y = y / norm;
	z = z / norm;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;


	glm::mat3 dL_dSigma = glm::mat3(
		dF_dcov3D[9* gs_idx + 0], 0.5f * dF_dcov3D[9* gs_idx + 1], 0.5f * dF_dcov3D[9* gs_idx + 2],
		0.5f * dF_dcov3D[9* gs_idx + 1], dF_dcov3D[9* gs_idx + 4], 0.5f * dF_dcov3D[9* gs_idx + 5],
		0.5f * dF_dcov3D[9* gs_idx + 2], 0.5f * dF_dcov3D[9* gs_idx + 5], dF_dcov3D[9* gs_idx + 8]
	);


	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	dF_dscale[gs_idx*3 + 0] = glm::dot(Rt[0], dL_dMt[0])*exp(scale_cuda[gs_idx*3 + 0]);
	dF_dscale[gs_idx*3 + 1] = glm::dot(Rt[1], dL_dMt[1])*exp(scale_cuda[gs_idx*3 + 1]);
	dF_dscale[gs_idx*3 + 2] = glm::dot(Rt[2], dL_dMt[2])*exp(scale_cuda[gs_idx*3 + 2]);

	// dF_dscale[gs_idx*3 + 0] = glm::dot(Rt[0], dL_dMt[0])*exp(scale_cuda[gs_idx*3 + 0])*(1.0f + ada_lpf_ratio[gs_idx*3 + 0]);
	// dF_dscale[gs_idx*3 + 1] = glm::dot(Rt[1], dL_dMt[1])*exp(scale_cuda[gs_idx*3 + 1])*(1.0f + ada_lpf_ratio[gs_idx*3 + 1]);
	// dF_dscale[gs_idx*3 + 2] = glm::dot(Rt[2], dL_dMt[2])*exp(scale_cuda[gs_idx*3 + 2])*(1.0f + ada_lpf_ratio[gs_idx*3 + 2]);


	dL_dMt[0] *= S[0][0];
	dL_dMt[1] *= S[1][1];
	dL_dMt[2] *= S[2][2];


	float dF_drot0 = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);

	float dF_drot1 = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);

	float dF_drot2 = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);

	float dF_drot3 = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	r = rot_cuda[gs_idx*4 + 0];
	x = rot_cuda[gs_idx*4 + 1];
	y = rot_cuda[gs_idx*4 + 2];
	z = rot_cuda[gs_idx*4 + 3];


	float divisor = pow(r*r + x*x + y*y + z*z, 3/2);
	dF_drot[gs_idx*4 + 0] = dF_drot0 * ((x*x + y*y + z*z)/divisor);
	dF_drot[gs_idx*4 + 0] -= dF_drot1 * ((r*x)/divisor);
	dF_drot[gs_idx*4 + 0] -= dF_drot2 * ((r*y)/divisor);
	dF_drot[gs_idx*4 + 0] -= dF_drot3 * ((r*z)/divisor);

	dF_drot[gs_idx*4 + 1] = dF_drot1 * ((r*r + y*y + z*z)/divisor);
	dF_drot[gs_idx*4 + 1] -= dF_drot0 * ((x*r)/divisor);
	dF_drot[gs_idx*4 + 1] -= dF_drot2 * ((x*y)/divisor);
	dF_drot[gs_idx*4 + 1] -= dF_drot3 * ((x*z)/divisor);

	dF_drot[gs_idx*4 + 2] = dF_drot2 * ((r*r + x*x + z*z)/divisor);
	dF_drot[gs_idx*4 + 2] -= dF_drot0 * ((y*r)/divisor);
	dF_drot[gs_idx*4 + 2] -= dF_drot1 * ((y*x)/divisor);
	dF_drot[gs_idx*4 + 2] -= dF_drot3 * ((y*z)/divisor);

	dF_drot[gs_idx*4 + 3] = dF_drot3 * ((r*r + x*x + y*y)/divisor);
	dF_drot[gs_idx*4 + 3] -= dF_drot0 * ((z*r)/divisor);
	dF_drot[gs_idx*4 + 3] -= dF_drot1 * ((z*x)/divisor);
	dF_drot[gs_idx*4 + 3] -= dF_drot2 * ((z*y)/divisor);

}


__global__ void computeGradsFromINVCov3D(
	int P, 
	const float* rot_cuda,
	const float* scale_cuda,
	const float* sigma_inv_cuda,
	const float* dF_dcov3D_inv,
	float* dF_drot,
	float* dF_dscale
	)
{
	int gs_idx = cg::this_grid().thread_rank();
	if (gs_idx >= P)
		return;
}


__device__ void computeGradsFromPDF(
	int gs_idx, 
	float cur_dF_dpdf,
	float log_pdf,
	const float* samples_pos,
	const float* A, // sigma_inv_cuda
	const float* pos_cuda,
	const float* rot_cuda,
	const float* scale_cuda,
	float* dF_dpos,
	float* dF_drot,
	float* dF_dscale,
	float* dF_dcov3D,
	float* ada_lpf_ratio
	)
{	
	float dF_dlogpdf = cur_dF_dpdf*(-0.5*exp(-0.5*log_pdf));

	float dF_dx =  dF_dlogpdf * 2 * (A[0]* (pos_cuda[0] - samples_pos[0]));
	dF_dx += dF_dlogpdf * 2 * (A[1] * (pos_cuda[1] - samples_pos[1]));
	dF_dx += dF_dlogpdf * 2 * (A[2] * (pos_cuda[2] - samples_pos[2]));

	float dF_dy =  dF_dlogpdf * 2 * (A[1]* (pos_cuda[0] - samples_pos[0]));
	dF_dy +=  dF_dlogpdf * 2 * (A[4]* (pos_cuda[1] - samples_pos[1]));
	dF_dy +=  dF_dlogpdf * 2 * (A[5]* (pos_cuda[2] - samples_pos[2]));
	
	float dF_dz =  dF_dlogpdf * 2 * (A[2]* (pos_cuda[0] - samples_pos[0]));
	dF_dz +=  dF_dlogpdf * 2 * (A[5]* (pos_cuda[1] - samples_pos[1]));
	dF_dz +=  dF_dlogpdf * 2 * (A[8]* (pos_cuda[2] - samples_pos[2]));


	atomicAdd(&dF_dpos[3* gs_idx + 0], dF_dx);
	atomicAdd(&dF_dpos[3* gs_idx + 1], dF_dy);
	atomicAdd(&dF_dpos[3* gs_idx + 2], dF_dz);

	float x = samples_pos[0] - pos_cuda[0];
	float y = samples_pos[1] - pos_cuda[1];
	float z = samples_pos[2] - pos_cuda[2];	

	glm::mat3 dF_dcov3D_inv = glm::mat3(
		dF_dlogpdf*x*x, dF_dlogpdf*x*y, dF_dlogpdf*x*z,
		dF_dlogpdf*x*y, dF_dlogpdf*y*y, dF_dlogpdf*y*z,
		dF_dlogpdf*x*z, dF_dlogpdf*y*z, dF_dlogpdf*z*z
	);

	glm::vec3 v = glm::vec3(x, y, z);
	glm::mat3 sigma_inv = glm::mat3(
		A[0], A[1], A[2],
		A[1], A[4], A[5],
		A[2], A[5], A[8]
	);

	glm::vec3 dpdf_dsigma0 = sigma_inv * v;
	glm::mat3 dpdf_dsigma1 = glm::outerProduct(dpdf_dsigma0, v);
	glm::mat3 dpdf_dsigma2 = dpdf_dsigma1 * sigma_inv;

	atomicAdd(&dF_dcov3D[9* gs_idx + 0], 0.5f * cur_dF_dpdf * exp(-0.5*log_pdf) * dpdf_dsigma2[0][0]);
	atomicAdd(&dF_dcov3D[9* gs_idx + 1], 0.5f * cur_dF_dpdf * exp(-0.5*log_pdf) * dpdf_dsigma2[0][1] * 2);
	atomicAdd(&dF_dcov3D[9* gs_idx + 2], 0.5f * cur_dF_dpdf * exp(-0.5*log_pdf) * dpdf_dsigma2[0][2] * 2);
	atomicAdd(&dF_dcov3D[9* gs_idx + 4], 0.5f * cur_dF_dpdf * exp(-0.5*log_pdf) * dpdf_dsigma2[1][1]);
	atomicAdd(&dF_dcov3D[9* gs_idx + 5], 0.5f * cur_dF_dpdf * exp(-0.5*log_pdf) * dpdf_dsigma2[1][2] * 2);
	atomicAdd(&dF_dcov3D[9* gs_idx + 8], 0.5f * cur_dF_dpdf * exp(-0.5*log_pdf) * dpdf_dsigma2[2][2]);

}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

__global__ void Clip3dGradsCUDA(
	const int N, int D, int M,
	const int P,
	float* dF_dopacity,
	float* dF_dshs,
	float* dF_drot,
	float* dF_dscale
){
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// for (int i = 0; i < D; i++){
	// 	dF_dshs[idx*D*M + i] = min(1.0f, dF_dshs[idx*D*M + i]);
	// 	dF_dshs[idx*D*M + i] = max(-1.0f, dF_dshs[idx*D*M + i]);
	// }

	// dF_dopacity[idx] = min(1.0f, dF_dopacity[idx]);
	// dF_dopacity[idx] = max(-1.0f, dF_dopacity[idx]);

	for (int i = 0; i < 4; i++){
		dF_drot[idx*4 + i] = min(1.0f, dF_drot[idx*D*M + i]);
		dF_drot[idx*4 + i] = max(-1.0f, dF_drot[idx*D*M + i]);
	}
}

__global__ void StepIncrement(
	int* step
){
	auto idx = cg::this_grid().thread_rank();
	if (idx >= 1)
		return;
	atomicAdd(&(step[0]), 1);
}

__global__ void CheckScaleCUDA(
	int P,
	float* scale_cuda,
	float* max_scale_cuda
){
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	
	for (int i = 0; i < 3; i++){
		if ((scale_cuda[idx*3 + i] > max_scale_cuda[idx*3 + i]) && (scale_cuda[idx*3 + i] > -2.0f)){
		// if ((scale_cuda[idx*3 + i] > max_scale_cuda[idx*3 + i])){
			scale_cuda[idx*3 + i] = max_scale_cuda[idx*3 + i];
			// printf("Scale Exceeded %f\n", scale_cuda[idx*3 + i]);
		}
	}
}

__global__ void CheckOpacityCUDA(
	int P,
	float* opacity_cuda
){
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	
	if (sigmoid(opacity_cuda[idx]) < 0.001f){
		opacity_cuda[idx] = 0.0;
	} 
}

__global__ void Updatefeature3dCUDA(
	const int P, int D, int M,
	const float* dF_dopacity,
	const float* dF_dshs,
	const float* dF_dpos,
	const float* dF_drot,
	const float* dF_dscale,
	float* opacity_cuda,
	float* shs_cuda,
	float* pos_cuda,
	float* rot_cuda,
	float* scale_cuda,
	float* m_opacity,
	float* v_opacity,
	float* m_shs,
	float* v_shs,
	float* m_pos,
	float* v_pos,
	float* m_rot,
	float* v_rot,
	float* m_scale,
	float* v_scale,
	float* max_scale_cuda,
	int* step,
	bool* opt_options_cuda,
	float* learning_rate_cuda,
	int _optimize_steps,
	int* moved_gaussians_cuda
)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (moved_gaussians_cuda[idx] == 0){
		return;
	}

	int shs_dim = D;
	if (opt_options_cuda[5]){
		shs_dim = D*M;
	}
	
	float beta1 = 0.9;
	float beta2 = 0.999;
	float epsilon = 1e-15;

	float t = min(((float)step[0]) / (float)100.0, 1.0);
    float lr_pos = exp(log(0.00032) * (1 - t) + log(0.0000064) * t);

	m_opacity[idx] = beta1*m_opacity[idx] + (1-beta1)*dF_dopacity[idx];
	v_opacity[idx] = beta2*v_opacity[idx] + (1-beta2)*dF_dopacity[idx]*dF_dopacity[idx];
	for (int i = 0; i < shs_dim; i++){
		m_shs[idx*D*M + i] = beta1*m_shs[idx*D*M + i] + (1-beta1)*dF_dshs[idx*D*M + i];
		v_shs[idx*D*M + i] = beta2*v_shs[idx*D*M + i] + (1-beta2)*dF_dshs[idx*D*M + i]*dF_dshs[idx*D*M + i];
	}
	for (int i = 0; i < 3; i++){
		m_pos[idx*3 + i] = beta1*m_pos[idx*3 + i] + (1-beta1)*dF_dpos[idx*3 + i];
		v_pos[idx*3 + i] = beta2*v_pos[idx*3 + i] + (1-beta2)*dF_dpos[idx*3 + i]*dF_dpos[idx*3 + i];
	}
	for (int i = 0; i < 4; i++){
		m_rot[idx*4 + i] = beta1*m_rot[idx*4 + i] + (1-beta1)*dF_drot[idx*4 + i];
		v_rot[idx*4 + i] = beta2*v_rot[idx*4 + i] + (1-beta2)*dF_drot[idx*4 + i]*dF_drot[idx*4 + i];
	}
	for (int i = 0; i < 3; i++){
		m_scale[idx*3 + i] = beta1*m_scale[idx*3 + i] + (1-beta1)*dF_dscale[idx*3 + i];
		v_scale[idx*3 + i] = beta2*v_scale[idx*3 + i] + (1-beta2)*dF_dscale[idx*3 + i]*dF_dscale[idx*3 + i];
	}	

	float mt_o = m_opacity[idx] / (1-pow(beta1, (float)step[0]));
	float vt_o = v_opacity[idx] / (1-pow(beta2, (float)step[0]));
	if (opt_options_cuda[3]){
		atomicAdd(&opacity_cuda[idx], -(learning_rate_cuda[3]*mt_o) / (sqrt(vt_o)+epsilon));
	}

	for (int i = 0; i < shs_dim; i++){
		float mt_s = m_shs[idx*D*M + i] / (1-pow(beta1, (float)step[0]));
		float vt_s = v_shs[idx*D*M + i] / (1-pow(beta2, (float)step[0]));
		if (opt_options_cuda[4]){
			atomicAdd(&shs_cuda[idx*D*M + i], -(learning_rate_cuda[4]*mt_s) / (sqrt(vt_s)+epsilon));
		}
	}

	for (int i = 0; i < 3; i++){
		float mt_pos = m_pos[idx*3 + i] / (1-pow(beta1, (float)step[0]));
		float vt_pos = v_pos[idx*3 + i] / (1-pow(beta2, (float)step[0]));
		if (opt_options_cuda[0]){
			atomicAdd(&pos_cuda[idx*3 + i], -(lr_pos*mt_pos) / (sqrt(vt_pos)+epsilon));
		}
	}

	for (int i = 0; i < 4; i++){
		float mt_rot = m_rot[idx*4 + i] / (1-pow(beta1, (float)step[0]));
		float vt_rot = v_rot[idx*4 + i] / (1-pow(beta2, (float)step[0]));
		if (opt_options_cuda[1]){
			atomicAdd(&rot_cuda[idx*4 + i], -(learning_rate_cuda[1]*mt_rot) / (sqrt(vt_rot)+epsilon));
		}
	}

	for (int i = 0; i < 3; i++){
		float mt_scale = m_scale[idx*3 + i] / (1-pow(beta1, (float)step[0]));
		float vt_scale = v_scale[idx*3 + i] / (1-pow(beta2, (float)step[0]));
		if (opt_options_cuda[2]){
			atomicAdd(&scale_cuda[idx*3 + i], -(learning_rate_cuda[2]*mt_scale) / (sqrt(vt_scale)+epsilon));
		}
	}

}

__global__ void compute3dgradsCUDA_grid(
	const int valid_grid_num, int D, int M,
	const int P, int S_PerGird,
	const int* valid_grid_cuda,
	const int* grid_gs_prefix_sum_cuda,
	const float* samples_pos,
	const float* pos_cuda,
	const float* rot_cuda,
	const float* scale_cuda,
	const float* opacity_cuda,
	const float* shs_cuda,
	const float* half_length_cuda,
	const float* sigma_inv_cuda,
	float* sigma_damp_cuda,
	const float* opacity_grad_cuda,
	const float* feature_grad_cuda,
	float* dF_dopacity,
	float* dF_dshs,
	float* dF_dpos,
	float* dF_drot,
	float* dF_dscale,
	float* dF_dcov3D,
	const int* grided_gs_idx_cuda,
	bool* grid_is_converged_cuda,
	bool* opt_options_cuda,
	float3 min_xyz,
	float grid_step,
	int grid_num,
	float* ada_lpf_ratio,
	int* empty_grid_cuda,
	int* current_static_grids_cuda,
	int* moved_gaussians_cuda,
	bool has_soup
	)
{
	if (grid_is_converged_cuda[blockIdx.x] && opt_options_cuda[11]){
		return;
	}

	int shs_dim = D;
	if (opt_options_cuda[5]){
		shs_dim = D*M;
	}

	auto idx = cg::this_grid().thread_rank();

	//
	if ((empty_grid_cuda[blockIdx.x] == 1) && opt_options_cuda[11]){
		return;
	}
	//
	int sample_idx_in_grid = threadIdx.x;
	
	int sp_idx = blockIdx.x * S_PerGird + sample_idx_in_grid;

	int x_idx = int(floor((samples_pos[sp_idx*3 + 0] - min_xyz.x)/grid_step));
	if ((x_idx < 0) || (x_idx >= grid_num)){
		return;
	}
	int y_idx = int(floor((samples_pos[sp_idx*3 + 1] - min_xyz.y)/grid_step));
	if ((y_idx < 0) || (y_idx >= grid_num)){
		return;
	}
	int z_idx = int(floor((samples_pos[sp_idx*3 + 2] - min_xyz.z)/grid_step));
	if ((z_idx < 0) || (z_idx >= grid_num)){
		return;
	}
	int cur_grid_idx = x_idx*grid_num*grid_num + y_idx*grid_num + z_idx;

	if ((current_static_grids_cuda[cur_grid_idx] == 1) && opt_options_cuda[11]){
		return;
	}

	int gs_idx_start = 0;
	int gs_idx_end = grid_gs_prefix_sum_cuda[cur_grid_idx];
	if (cur_grid_idx != 0){
		gs_idx_start = grid_gs_prefix_sum_cuda[cur_grid_idx-1];
	}

	float alpha_threshold = 1.0f/255.0f;
	if (has_soup){
		alpha_threshold = 1.0f/1000.0f;
	}


	for (int i = gs_idx_start; i < gs_idx_end; i++){
		int gs_idx = grided_gs_idx_cuda[i];
		if (moved_gaussians_cuda[gs_idx] == 0){
			continue;
		}

		float x = samples_pos[sp_idx*3 + 0] - pos_cuda[gs_idx*3 + 0];
		float y = samples_pos[sp_idx*3 + 1] - pos_cuda[gs_idx*3 + 1];
		float z = samples_pos[sp_idx*3 + 2] - pos_cuda[gs_idx*3 + 2];

		float log_pdf = 0.0;
		log_pdf += sigma_inv_cuda[9*gs_idx + 0] * x * x;
		log_pdf += sigma_inv_cuda[9*gs_idx + 1] * x * y * 2;
		log_pdf += sigma_inv_cuda[9*gs_idx + 2] * x * z * 2;
		log_pdf += sigma_inv_cuda[9*gs_idx + 4] * y * y;
		log_pdf += sigma_inv_cuda[9*gs_idx + 5] * y * z * 2;
		log_pdf += sigma_inv_cuda[9*gs_idx + 8] * z * z;

		float cur_pdf;

		cur_pdf = exp(-0.5 *log_pdf);

		float sigmoid_opa = 1.0f / (1.0f + exp(-opacity_cuda[gs_idx]));
		if (cur_pdf * sigmoid_opa < alpha_threshold){
			continue;
		}
		float d_sigmoid_opa = (exp(-opacity_cuda[gs_idx])) / ((1.0f + exp(-opacity_cuda[gs_idx])) * (1.0f + exp(-opacity_cuda[gs_idx])));


		float cur_dF_do = 0.0f;
		float cur_dF_dshs = cur_pdf*sigmoid_opa;
		float cur_dF_dpdf = 0.0f;
		for (int j = 0; j < shs_dim; j++){
			cur_dF_do += shs_cuda[D*M*gs_idx + j] * feature_grad_cuda[D*M*sp_idx + j];
			cur_dF_dpdf += shs_cuda[D*M*gs_idx + j] * feature_grad_cuda[D*M*sp_idx + j];
		}
		// opacity loss
		cur_dF_do += opacity_grad_cuda[sp_idx];
		cur_dF_dpdf += opacity_grad_cuda[sp_idx];
		//
		cur_dF_do = d_sigmoid_opa * cur_pdf * cur_dF_do;
		cur_dF_dpdf = sigmoid_opa * cur_dF_dpdf;

		for (int j = 0; j < shs_dim; j++){
			atomicAdd(&dF_dshs[D*M * gs_idx + j], cur_dF_dshs*feature_grad_cuda[D*M*sp_idx + j]);
		}
		atomicAdd(&dF_dopacity[gs_idx], cur_dF_do);


		// Calculate the dF_dpos, dF_drot, and dF_dscale from dF_dpdf
		if (cur_dF_dpdf != 0.0) {
			computeGradsFromPDF(gs_idx, cur_dF_dpdf, log_pdf, samples_pos+sp_idx*3, sigma_inv_cuda+gs_idx*9, pos_cuda+gs_idx*3, rot_cuda+gs_idx*4, scale_cuda+gs_idx*3, dF_dpos, dF_drot, dF_dscale, dF_dcov3D, ada_lpf_ratio);
		}
	}

}


__global__ void compute3dgradsCUDA(
	const int N, int D, int M,
	const int P,
	const float* samples_pos,
	const int* sample_neighbours,
	const int* sample_idx_itselves,
	const float* pos_cuda,
	const float* rot_cuda,
	const float* scale_cuda,
	const float* opacity_cuda,
	const float* shs_cuda,
	const float* half_length_cuda,
	const float* sigma_inv_cuda,
	const float* feature_grad_cuda,
	float* dF_dopacity,
	float* dF_dshs,
	float* dF_dpos,
	float* dF_drot,
	float* dF_dscale,
	float* dF_dcov3D,
	float* ada_lpf_ratio
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
		return;
	
	int sp_idx = sample_idx_itselves[idx];
	int gs_idx = sample_neighbours[idx];

	float x = samples_pos[sp_idx*3 + 0] - pos_cuda[gs_idx*3 + 0];
	float y = samples_pos[sp_idx*3 + 1] - pos_cuda[gs_idx*3 + 1];
	float z = samples_pos[sp_idx*3 + 2] - pos_cuda[gs_idx*3 + 2];

	float log_pdf = 0.0;
	log_pdf += sigma_inv_cuda[9*gs_idx + 0] * x * x;
	log_pdf += sigma_inv_cuda[9*gs_idx + 1] * x * y * 2;
	log_pdf += sigma_inv_cuda[9*gs_idx + 2] * x * z * 2;
	log_pdf += sigma_inv_cuda[9*gs_idx + 4] * y * y;
	log_pdf += sigma_inv_cuda[9*gs_idx + 5] * y * z * 2;
	log_pdf += sigma_inv_cuda[9*gs_idx + 8] * z * z;

	float cur_pdf;

	if (-0.5 *log_pdf < -30){
		cur_pdf = 0.0;
	}
	else {
		cur_pdf = exp(-0.5 *log_pdf); // here the error occurs
	}


	float shs_r = shs_cuda[3*M*gs_idx + 0];
	float shs_g = shs_cuda[3*M*gs_idx + 1];
	float shs_b = shs_cuda[3*M*gs_idx + 2];

	float sign_r = feature_grad_cuda[3*M*sp_idx + 0];
	float sign_g = feature_grad_cuda[3*M*sp_idx + 1];
	float sign_b = feature_grad_cuda[3*M*sp_idx + 2];

	float sigmoid_opa = 1.0f / (1.0f + exp(-opacity_cuda[gs_idx]));;
	float d_sigmoid_opa = (exp(-opacity_cuda[gs_idx])) / ((1.0f + exp(-opacity_cuda[gs_idx])) * (1.0f + exp(-opacity_cuda[gs_idx])));

	float cur_dF_do = d_sigmoid_opa*cur_pdf*(shs_r*sign_r + shs_g*sign_g + shs_b*sign_b);
	float cur_dF_dshs = cur_pdf*sigmoid_opa;
	float cur_dF_dpdf = sigmoid_opa*(shs_r*sign_r + shs_g*sign_g + shs_b*sign_b);


	if (cur_pdf * sigmoid_opa >= 0.999){
		cur_dF_do = 0.0;
		cur_dF_dshs = 0.999;
		cur_dF_dpdf = 0.0;
	}
	else if (cur_pdf * sigmoid_opa < 0.01){
		cur_dF_do = 0.0;
		cur_dF_dshs = 0.0;
		cur_dF_dpdf = 0.0;
	}

	atomicAdd(&dF_dopacity[gs_idx], cur_dF_do);

	atomicAdd(&dF_dshs[3*M * gs_idx + 0], cur_dF_dshs*sign_r);
	atomicAdd(&dF_dshs[3*M * gs_idx + 1], cur_dF_dshs*sign_g);
	atomicAdd(&dF_dshs[3*M * gs_idx + 2], cur_dF_dshs*sign_b);

	// Calculate the dF_dpos, dF_drot, and dF_dscale from dF_dpdf
	if (cur_dF_dpdf != 0.0) {
		computeGradsFromPDF(gs_idx, cur_dF_dpdf, log_pdf, samples_pos+sp_idx*3, sigma_inv_cuda+gs_idx*9, pos_cuda+gs_idx*3, rot_cuda+gs_idx*4, scale_cuda+gs_idx*3, dF_dpos, dF_drot, dF_dscale, dF_dcov3D, ada_lpf_ratio);
	}

}



void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors
		);
}

void BACKWARD::compute3dgrads_grid(
	const int valid_grid_num, int D, int M,
	const int P, int S_PerGird,
	const int* valid_grid_cuda,
	const int* grid_gs_prefix_sum_cuda,
	const float* samples_pos,
	const float* pos_cuda,
	const float* rot_cuda,
	const float* scale_cuda,
	const float* opacity_cuda,
	const float* shs_cuda,
	const float* half_length_cuda,
	const float* sigma_cuda,
	float* sigma_damp_cuda,
	const float* opacity_grad_cuda,
	const float* feature_grad_cuda,
	float* dF_dopacity,
	float* dF_dshs,
	float* dF_dpos,
	float* dF_drot,
	float* dF_dscale,
	float* dF_dcov3D,
	const int* grided_gs_idx_cuda,
	bool* grid_is_converged_cuda,
	bool* opt_options_cuda,
	float3 min_xyz,
	float grid_step,
	int grid_num,
	float* ada_lpf_ratio,
	int* empty_grid_cuda,
	int* current_static_grids_cuda,
	int* moved_gaussians_cuda,
	bool has_soup
){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop) ;
	cudaEventRecord(start, 0);

	Init3dGradsCUDA <<< (P*D*M + 255) / 256, 256 >>>(
		P, D, M,
		dF_dopacity,
		dF_dshs,
		dF_dpos,
		dF_drot,
		dF_dscale,
		dF_dcov3D
	);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// printf("Time to initial 3d grads:  %3.1f ms\n", elapsedTime);

	compute3dgradsCUDA_grid<< <valid_grid_num, S_PerGird>> > (
		valid_grid_num, D, M,
		P, S_PerGird,
		valid_grid_cuda,
		grid_gs_prefix_sum_cuda,
		samples_pos,
		pos_cuda,
		rot_cuda,
		scale_cuda,
		opacity_cuda,
		shs_cuda,
		half_length_cuda,
		sigma_cuda,
		sigma_damp_cuda,
		opacity_grad_cuda,
		feature_grad_cuda,
		dF_dopacity,
		dF_dshs,
		dF_dpos,
		dF_drot,
		dF_dscale,
		dF_dcov3D,
		grided_gs_idx_cuda,
		grid_is_converged_cuda,
		opt_options_cuda,
		min_xyz,
		grid_step,
		grid_num,
		ada_lpf_ratio,
		empty_grid_cuda,
		current_static_grids_cuda,
		moved_gaussians_cuda,
		has_soup
	);
	cudaDeviceSynchronize();

	cudaEventRecord(start, 0) ;
	cudaEventSynchronize(start);
	cudaEventElapsedTime(&elapsedTime, stop, start);
	printf("Time to compute 3d grads:  %3.1f ms\n", elapsedTime);

	// computeGradsForAdaCov3D <<<(P + 255) / 256, 256>>>(
	// 	P,
	// 	dF_dcov3D,
	// 	ada_lpf_ratio
	// );
	// cudaDeviceSynchronize();

	computeGradsFromCov3D << <(P + 255) / 256, 256 >> >(
		P, 
		rot_cuda,
		scale_cuda,
		dF_dcov3D,
		dF_drot,
		dF_dscale,
		ada_lpf_ratio,
		opt_options_cuda
	);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// printf("Time to compute 3d grads from covariance:  %3.1f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void BACKWARD::compute3dgrads( // DO NOT USE THIS IMPLEMENTATION
	const int N, int D, int M,
	const int P,
	const float* samples_pos,
	const int* sample_neighbours,
	const int* sample_idx_itselves,
	const float* pos_cuda,
	const float* rot_cuda,
	const float* scale_cuda,
	const float* opacity_cuda,
	const float* shs_cuda,
	const float* half_length_cuda,
	const float* sigma_cuda,
	const float* feature_grad_cuda,
	float* dF_dopacity,
	float* dF_dshs,
	float* dF_dpos,
	float* dF_drot,
	float* dF_dscale,
	float* dF_dcov3D
){
	Init3dGradsCUDA <<< (P*D*M + 255) / 256, 256 >>>(
		P, D, M,
		dF_dopacity,
		dF_dshs,
		dF_dpos,
		dF_drot,
		dF_dscale,
		dF_dcov3D
	);
	cudaDeviceSynchronize();


	compute3dgradsCUDA<< <(N + 255) / 256, 256 >> > (
		N, D, M,
		P,
		samples_pos,
		sample_neighbours,
		sample_idx_itselves,
		pos_cuda,
		rot_cuda,
		scale_cuda,
		opacity_cuda,
		shs_cuda,
		half_length_cuda,
		sigma_cuda,
		feature_grad_cuda,
		dF_dopacity,
		dF_dshs,
		dF_dpos,
		dF_drot,
		dF_dscale,
		dF_dcov3D,
		NULL
	);
	cudaDeviceSynchronize();

	computeGradsFromCov3D << <(P + 255) / 256, 256 >> >(
		P, 
		rot_cuda,
		scale_cuda,
		dF_dcov3D,
		dF_drot,
		dF_dscale,
		NULL,
		NULL
	);
	cudaDeviceSynchronize();
	// Clip3dGradsCUDA <<< (P + 255) / 256, 256 >>>(
	// 	N, D, M,
	// 	P,
	// 	dF_dopacity,
	// 	dF_dshs,
	// 	dF_drot,
	// 	dF_dscale
	// );
	// cudaDeviceSynchronize();
}


void BACKWARD::updatefeature3d(
	const int P, int D, int M,
	const float* dF_dopacity,
	const float* dF_dshs,
	const float* dF_dpos,
	const float* dF_drot,
	const float* dF_dscale,
	float* opacity_cuda,
	float* shs_cuda,
	float* pos_cuda,
	float* rot_cuda,
	float* scale_cuda,
	float* m_opacity_cuda,
	float* v_opacity_cuda,
	float* m_shs_cuda,
	float* v_shs_cuda,
	float* m_pos_cuda,
	float* v_pos_cuda,
	float* m_rot_cuda,
	float* v_rot_cuda,
	float* m_scale_cuda,
	float* v_scale_cuda,
	float* max_scale_cuda,
	int* step,
	bool* opt_options_cuda,
	float* learning_rate_cuda,
	int _optimize_steps,
	int* moved_gaussians_cuda
	)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop) ;
	cudaEventRecord(start, 0);

	StepIncrement <<< 1, 256 >>>(
		step
	);
	cudaDeviceSynchronize();

	Updatefeature3dCUDA <<< (P + 255) / 256, 256 >>>(
		P, D, M,
		dF_dopacity,
		dF_dshs,
		dF_dpos,
		dF_drot,
		dF_dscale,
		opacity_cuda,
		shs_cuda,
		pos_cuda,
		rot_cuda,
		scale_cuda,
		m_opacity_cuda,
		v_opacity_cuda,
		m_shs_cuda,
		v_shs_cuda,
		m_pos_cuda,
		v_pos_cuda,
		m_rot_cuda,
		v_rot_cuda,
		m_scale_cuda,
		v_scale_cuda,
		max_scale_cuda,
		step,
		opt_options_cuda,
		learning_rate_cuda,
		_optimize_steps,
		moved_gaussians_cuda
	);
	cudaDeviceSynchronize();

	// CheckScaleCUDA <<< (P + 255) / 256, 256 >>>(
	// 	P,
	// 	scale_cuda,
	// 	max_scale_cuda
	// );
	// cudaDeviceSynchronize();

	// CheckOpacityCUDA <<< (P + 255) / 256, 256 >>>(
	// 	P,
	// 	opacity_cuda
	// );
	// cudaDeviceSynchronize();

	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to update 3d features:  %3.1f ms\n", elapsedTime);

}


