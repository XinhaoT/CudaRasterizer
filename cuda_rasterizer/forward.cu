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
#include <stdio.h>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix, const float* projmatrix, bool _to_ortho, float ortho_scale)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// printf("J: %f %f %f %f", J[0][0], J[0][2], J[1][1], J[1][2]);
	
	if (_to_ortho){
		J = glm::mat3(
			400.0f / ortho_scale, 		0.0f,		0.0f,				
			0.0f,		400.0f / ortho_scale,		0.0f,			
			0.0f,		0.0f,		0.0f);
	}

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return;
}

__global__ void judge_convergencyCUDA(
	int S_PerGird,
	int valid_grid_num,
	bool* grid_is_converged_cuda,
	bool* grid_nearly_converged_cuda,
	float* grid_loss_sums_cuda,
	bool has_soup
){
	auto idx = cg::this_grid().thread_rank();
	if (idx > valid_grid_num){
		return;
	}

	float alpha_threshold = 1.0f/255.0f;
	if (has_soup){
		alpha_threshold = 1.0f/1000.0f;
	}

	if (grid_loss_sums_cuda[idx] < (alpha_threshold * 48)){
		grid_is_converged_cuda[idx] = true;
	}
	else {
		grid_is_converged_cuda[idx] = false;
	}

}

__global__ void compute3dlossCUDA(
	int D, int M,
	const int S,
	const float* aim_feature_cuda,
	const float* cur_feature_cuda,
	const float* aim_opacity_cuda,
	const float* cur_opacity_cuda,
	float* opacity_grad_cuda,
	float* feature_grad_cuda,
	float* total_feature_loss,
	float* total_shape_loss,
	bool* grid_is_converged_cuda,
	bool* grid_nearly_converged_cuda,
	float* grid_loss_sums_cuda,
	bool* opt_options_cuda
	)
{
	int grid_idx = blockIdx.x;
	if (grid_is_converged_cuda[grid_idx] && opt_options_cuda[11]){
		return;
	}

	int shs_dim = D;
	if (opt_options_cuda[5]){
		shs_dim = D*M;
	}

	float cur_sample_loss = 0.0f;

	auto idx = cg::this_grid().thread_rank();

	for (int i = 0; i < shs_dim; i++){	//Only the 0-th SHs
		int offset = idx*D*M + i;
		float loss_per_dim = aim_feature_cuda[offset] - cur_feature_cuda[offset];

		
		if (loss_per_dim > 1e-8){
			feature_grad_cuda[offset] = -1.0;
		}
		else if (loss_per_dim < -1e-8){
			feature_grad_cuda[offset] = 1.0;
		}
		else{
			feature_grad_cuda[offset] = 0.0;
		}
		
		
		if (opt_options_cuda[6]) {
			atomicAdd(&total_feature_loss[0], abs(loss_per_dim));
		}

		// atomicAdd(&grid_loss_sums_cuda[grid_idx], abs(loss_per_dim));
		// atomicMaxFloat(&grid_loss_sums_cuda[grid_idx], abs(loss_per_dim));
		cur_sample_loss += abs(loss_per_dim);
	}


	if (opt_options_cuda[8]) {
		float opacity_loss = aim_opacity_cuda[idx] - cur_opacity_cuda[idx];
		// atomicAdd(&grid_loss_sums_cuda[grid_idx], abs(opacity_loss));
		// cur_sample_loss += abs(opacity_loss);
		
		if (opacity_loss > 1e-8){
			opacity_grad_cuda[idx] = -1.0;
		}
		else if (opacity_loss < -1e-8){
			opacity_grad_cuda[idx] = 1.0;
		}
		else{
			opacity_grad_cuda[idx] = 0.0;
		}


		// L0 loss 
		if (opt_options_cuda[7]) {
			if ((aim_opacity_cuda[idx] == 0.0f) || (aim_opacity_cuda[idx] == 0.0f)) {
				opacity_grad_cuda[idx] = opacity_grad_cuda[idx] * 100.0;
				// atomicAdd(&grid_loss_sums_cuda[grid_idx], abs(opacity_loss)*99.0f);
				if (opt_options_cuda[10]) {
					atomicAdd(&total_shape_loss[0], abs(opacity_loss) * 99.0f);
				}
			}
		}

		if (opt_options_cuda[10]) {
			atomicAdd(&total_shape_loss[0], abs(opacity_loss));
		}

	}

	atomicMaxFloat(&grid_loss_sums_cuda[grid_idx], cur_sample_loss);

}


__global__ void compute3dcovarianceCUDA(
	const int P,
	const int grid_num,
	const float grid_step,
	float3 min_xyz,
	const float* pos_cuda,
	const float* rot_cuda,
	const float* scale_cuda,
	float* sigma_inv_cuda,
	float* sigma_damp_cuda,
	float low_pass_param,
	float* ada_lpf_ratio,
	bool* opt_options_cuda,
	int* gs_init_grid_idx_cuda
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	int cur_grid_idx = gs_init_grid_idx_cuda[idx];

	float r = rot_cuda[idx*4 + 0];
	float x = rot_cuda[idx*4 + 1];
	float y = rot_cuda[idx*4 + 2];
	float z = rot_cuda[idx*4 + 3];
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

	glm::mat3 S = glm::mat3(1.0f);

	S[0][0] = exp(scale_cuda[idx*3 + 0])+0.0f;
	S[1][1] = exp(scale_cuda[idx*3 + 1])+0.0f;
	S[2][2] = exp(scale_cuda[idx*3 + 2])+0.0f;


	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	if (opt_options_cuda[9]){ // low-pass filter
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				Sigma[i][j] += ada_lpf_ratio[cur_grid_idx*9 + i*3 + j];
			}
		}
	}
	else{
		Sigma[0][0] += 5e-7; // only for the precision error in ablation study of low-pass filter
		Sigma[1][1] += 5e-7; // in the ship case, the value should increase to 5e-6 to avoid the numerical problems
		Sigma[2][2] += 5e-7;
	}


	float detSigma = Sigma[0][0] * (Sigma[1][1] * Sigma[2][2] - Sigma[1][2] * Sigma[2][1]) + 
					 Sigma[0][1] * (Sigma[1][2] * Sigma[2][0] - Sigma[1][0] * Sigma[2][2]) + 
					 Sigma[0][2] * (Sigma[1][0] * Sigma[2][1] - Sigma[1][1] * Sigma[2][0]);

	if (detSigma == 0.0f) {
		detSigma = 0.0000001f;
	}

	// Low pass filter added here
	sigma_inv_cuda[idx*9 + 0] = (Sigma[1][1] * Sigma[2][2] - Sigma[1][2] * Sigma[2][1]) / detSigma;
	sigma_inv_cuda[idx*9 + 1] = (Sigma[1][2] * Sigma[2][0] - Sigma[1][0] * Sigma[2][2]) / detSigma;
	sigma_inv_cuda[idx*9 + 2] = (Sigma[1][0] * Sigma[2][1] - Sigma[1][1] * Sigma[2][0]) / detSigma;
	sigma_inv_cuda[idx*9 + 3] = (Sigma[2][1] * Sigma[0][2] - Sigma[2][2] * Sigma[0][1]) / detSigma;
	sigma_inv_cuda[idx*9 + 4] = (Sigma[2][2] * Sigma[0][0] - Sigma[2][0] * Sigma[0][2]) / detSigma;
	sigma_inv_cuda[idx*9 + 5] = (Sigma[2][0] * Sigma[0][1] - Sigma[2][1] * Sigma[0][0]) / detSigma;
	sigma_inv_cuda[idx*9 + 6] = (Sigma[0][1] * Sigma[1][2] - Sigma[0][2] * Sigma[1][1]) / detSigma;
	sigma_inv_cuda[idx*9 + 7] = (Sigma[0][2] * Sigma[1][0] - Sigma[0][0] * Sigma[1][2]) / detSigma;
	sigma_inv_cuda[idx*9 + 8] = (Sigma[0][0] * Sigma[1][1] - Sigma[0][1] * Sigma[1][0]) / detSigma;

}


__global__ void compute3dfeaturesCUDA_grid(
	const int valid_grid_num, int D, int M,
	const int P, int S_PerGird,
	const int* valid_grid_cuda,
	const int* grid_gs_prefix_sum_cuda,
	const float* samples_pos,
	const float* pos_cuda,
	float* rot_cuda,
	const float* scale_cuda,
	const float* opacity_cuda,
	const float* shs_cuda,
	const float* half_length_cuda,
	float* sigma_inv_cuda,
	float* sigma_damp_cuda,
	float* result,
	float* feature_opacity_cuda,
	const int* grided_gs_idx_cuda,
	bool* grid_is_converged_cuda,
	bool* opt_options_cuda,
	float3 min_xyz,
	float grid_step,
	float grid_num,
	int* empty_grid_cuda,
	int* current_static_grids_cuda,
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

		float non_transparent;
		if ((cur_pdf * sigmoid_opa < alpha_threshold)){
			continue;
		}
		else {
			non_transparent = cur_pdf * sigmoid_opa;
		}

		atomicAdd(&feature_opacity_cuda[sp_idx], non_transparent);

		for (int j = 0; j < shs_dim; j++) {
			atomicAdd(&result[D*M * sp_idx + j], non_transparent * shs_cuda[D*M*gs_idx + j]);
		}
	}
}


// __global__ void compute3dfeaturesCUDA_grid_shared(
// 	const int valid_grid_num, int D, int M,
// 	const int P, int S_PerGird,
// 	const int* valid_grid_cuda,
// 	const int* grid_gs_prefix_sum_cuda,
// 	const float* samples_pos,
// 	const float* pos_cuda,
// 	float* rot_cuda,
// 	const float* scale_cuda,
// 	const float* opacity_cuda,
// 	const float* shs_cuda,
// 	const float* half_length_cuda,
// 	float* sigma_inv_cuda,
// 	float* result,
// 	float* feature_opacity_cuda,
// 	const int* grided_gs_idx_cuda,
// 	bool* grid_is_converged_cuda,
// 	bool* opt_options_cuda
// 	)
// {
// 	if (grid_is_converged_cuda[blockIdx.x]){
// 		return;
// 	}

// 	int shs_dim = D;
// 	if (opt_options_cuda[5]){
// 		shs_dim = D*M;
// 	}

// 	auto block = cg::this_thread_block();
// 	auto idx = cg::this_grid().thread_rank();
// 	int grid_idx = valid_grid_cuda[blockIdx.x];
// 	int sample_idx_in_grid = threadIdx.x;
	
// 	int sp_idx = blockIdx.x * S_PerGird + sample_idx_in_grid;
// 	float sp_x = samples_pos[sp_idx*3 + 0];
// 	float sp_y = samples_pos[sp_idx*3 + 1];
// 	float sp_z = samples_pos[sp_idx*3 + 2];

// 	int gs_idx_start = 0;
// 	int gs_idx_end = grid_gs_prefix_sum_cuda[grid_idx];
// 	if (grid_idx != 0){
// 		gs_idx_start = grid_gs_prefix_sum_cuda[grid_idx-1];
// 	}

// 	int toDo = gs_idx_end - gs_idx_start;
// 	int rounds = (toDo + BLOCK_SIZE - 1) / BLOCK_SIZE;

// 	__shared__ int collected_gs_idx[BLOCK_SIZE];
// 	__shared__ float3 collected_gs_pos[BLOCK_SIZE];
// 	__shared__ float collected_gs_opacity[BLOCK_SIZE];
// 	__shared__ float3 collected_gs_sigma_first_row[BLOCK_SIZE];
// 	__shared__ float3 collected_gs_sigma_second_row[BLOCK_SIZE];

// 	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
// 		int progress = i * BLOCK_SIZE + block.thread_rank();
// 		if (gs_idx_start + progress < gs_idx_end) {
// 			int gs_idx = grided_gs_idx_cuda[gs_idx_start + progress];
// 			collected_gs_idx[block.thread_rank()] = gs_idx;

// 			collected_gs_pos[block.thread_rank()].x = pos_cuda[gs_idx*3 + 0];
// 			collected_gs_pos[block.thread_rank()].y = pos_cuda[gs_idx*3 + 1];
// 			collected_gs_pos[block.thread_rank()].z = pos_cuda[gs_idx*3 + 2];

// 			collected_gs_opacity[block.thread_rank()] = opacity_cuda[gs_idx];

// 			collected_gs_sigma_first_row[block.thread_rank()].x = sigma_inv_cuda[9*gs_idx + 0];
// 			collected_gs_sigma_first_row[block.thread_rank()].y = sigma_inv_cuda[9*gs_idx + 1];
// 			collected_gs_sigma_first_row[block.thread_rank()].z = sigma_inv_cuda[9*gs_idx + 2];

// 			collected_gs_sigma_second_row[block.thread_rank()].x = sigma_inv_cuda[9*gs_idx + 4];
// 			collected_gs_sigma_second_row[block.thread_rank()].y = sigma_inv_cuda[9*gs_idx + 5];
// 			collected_gs_sigma_second_row[block.thread_rank()].z = sigma_inv_cuda[9*gs_idx + 8];
// 		}
// 		block.sync();

		// for (int j = 0; j < min(BLOCK_SIZE, toDo); j++){
		// 	float x = sp_x - collected_gs_pos[j].x;
		// 	float y = sp_y - collected_gs_pos[j].y;
		// 	float z = sp_z - collected_gs_pos[j].z;

		// 	float log_pdf = 0.0;
		// 	log_pdf += collected_gs_sigma_first_row[j].x * x * x;
		// 	log_pdf += collected_gs_sigma_first_row[j].y * x * y * 2;
		// 	log_pdf += collected_gs_sigma_first_row[j].z * x * z * 2;
		// 	log_pdf += collected_gs_sigma_second_row[j].x * y * y;
		// 	log_pdf += collected_gs_sigma_second_row[j].y * y * z * 2;
		// 	log_pdf += collected_gs_sigma_second_row[j].z * z * z;

		// 	float cur_pdf;
		// 	if ((-0.5 *log_pdf < -7.0f)) {
		// 		cur_pdf = 0.0;
		// 		continue;
		// 	}
		// 	else {
		// 		cur_pdf = exp(-0.5 *log_pdf);
		// 	}

		// 	float sigmoid_opa = 1.0f / (1.0f + exp(-collected_gs_opacity[j]));

		// 	float non_transparent;
		// 	if ((cur_pdf * sigmoid_opa < 0.01)){
		// 		non_transparent = 0.0;
		// 		continue;
		// 	}
		// 	else if (cur_pdf * sigmoid_opa >= 0.999){
		// 		non_transparent = 0.999;
		// 	}
		// 	else {
		// 		non_transparent = cur_pdf * sigmoid_opa;
		// 	}

		// 	atomicAdd(&feature_opacity_cuda[sp_idx], non_transparent);

		// 	for (int k = 0; k < shs_dim; k++) {
		// 		// atomicAdd(&result[shs_dim* sp_idx + k], non_transparent * shs_cuda[shs_dim*collected_gs_idx[j] + k]);
		// 		// atomicAdd(&result[shs_dim* sp_idx + k], non_transparent);
		// 	}
		// }

// 	}
// }


__global__ void compute3dfeaturesCUDA(
	const int N, int D, int M,
	const int S,
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
	float* result,
	float* feature_opacity_cuda
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

	if ((-0.5 *log_pdf < -30.0f)) {
		cur_pdf = 0.0;
	}
	else {
		cur_pdf = exp(-0.5 *log_pdf);
	}


	float sigmoid_opa = 1.0f / (1.0f + exp(-opacity_cuda[gs_idx]));

	float non_transparent;
	if (cur_pdf * sigmoid_opa >= 0.999){
		non_transparent = 0.999;
	}
	else if ((cur_pdf * sigmoid_opa < 0.01)){
		non_transparent = 0.0;
	}
	else {
		non_transparent = cur_pdf * sigmoid_opa;
	}

	float shs_r = non_transparent * shs_cuda[3*M*gs_idx + 0];
	float shs_g = non_transparent * shs_cuda[3*M*gs_idx + 1];
	float shs_b = non_transparent * shs_cuda[3*M*gs_idx + 2];

	atomicAdd(&feature_opacity_cuda[sp_idx], non_transparent);

	atomicAdd(&result[3*M * sp_idx + 0], shs_r);
	atomicAdd(&result[3*M * sp_idx + 1], shs_g);
	atomicAdd(&result[3*M * sp_idx + 2], shs_b);
}

__global__ void Init3dfeaturesCUDA(
	const int total_dim,
	float* result
	)
{
	int idx = cg::this_grid().thread_rank();
	if (idx < total_dim) {
		result[idx] = 0.0;
	}

}


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	int2* rects,
	float3 boxmin,
	float3 boxmax,
	bool _to_ortho,
	float ortho_scale)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	if (p_orig.x < boxmin.x || p_orig.y < boxmin.y || p_orig.z < boxmin.z ||
		p_orig.x > boxmax.x || p_orig.y > boxmax.y || p_orig.z > boxmax.z)
		return;

	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, projmatrix, _to_ortho, ortho_scale);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;

	if (rects == nullptr) 	// More conservative
	{
		getRect(point_image, my_radius, rect_min, rect_max, grid);
	}
	else // Slightly more aggressive, might need a math cleanup
	{
		const int2 my_rect = { (int)ceil(3.f * sqrt(cov.x)), (int)ceil(3.f * sqrt(cov.z)) };
		rects[idx] = my_rect;
		getRect(point_image, my_rect, rect_min, rect_max, grid);
	}

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int* __restrict__ is_surface)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) // modified to avoid numerical 
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			// Fetch the Surface Gaussians
			// if (T > 0.999f){
			// 	atomicAdd(&is_surface[collected_id[j]], 1);
			// }

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	int* is_surface)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		is_surface);
}

void FORWARD::compute3dfeatures_grid(
	const int valid_grid_num, int D, int M,
	const int P, int S_PerGird,
	const int* valid_grid_cuda,
	const int* grid_gs_prefix_sum_cuda,
	const float* samples_pos,
	const float* pos_cuda,
	float* rot_cuda,
	const float* scale_cuda,
	const float* opacity_cuda,
	const float* shs_cuda,
	const float* half_length_cuda,
	float* sigma_cuda,
	float* sigma_damp_cuda,
	float* result,
	float* feature_opacity_cuda,
	const int* grided_gs_idx_cuda,
	bool* grid_is_converged_cuda,
	bool* grid_nearly_converged_cuda,
	bool* opt_options_cuda,
	float low_pass_param,
	float* ada_lpf_ratio,
	float3 min_xyz,
	float grid_step,
	int grid_num,
	int* gs_init_grid_idx_cuda,
	int* empty_grid_cuda,
	int* current_static_grids_cuda,
	bool has_soup
	)
{
	int S = valid_grid_num * S_PerGird;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop) ;
	cudaEventRecord(start, 0) ;

	cudaDeviceSynchronize();
	Init3dfeaturesCUDA <<< (S*M*D + 255) / 256, 256 >>>(
		S*M*D,
		result
	);
	cudaDeviceSynchronize();
	Init3dfeaturesCUDA <<< (S + 255) / 256, 256 >>>(
		S,
		feature_opacity_cuda
	);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// printf("Time to initial 3d features:  %3.1f ms\n", elapsedTime);

	compute3dcovarianceCUDA<< <(P + 255) / 256, 256 >> >(
		P,
		grid_num,
		grid_step,
		min_xyz,
		pos_cuda,
		rot_cuda,
		scale_cuda,
		sigma_cuda,
		sigma_damp_cuda,
		low_pass_param,
		ada_lpf_ratio,
		opt_options_cuda,
		gs_init_grid_idx_cuda
	);
	cudaDeviceSynchronize();

	cudaEventRecord(start, 0) ;
	cudaEventSynchronize(start);
	cudaEventElapsedTime(&elapsedTime, stop, start);
	// printf("Time to compute 3d covariance:  %3.1f ms\n", elapsedTime);
	

	compute3dfeaturesCUDA_grid<<<valid_grid_num, S_PerGird>>>(
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
		result,
		feature_opacity_cuda,
		grided_gs_idx_cuda,
		grid_is_converged_cuda,
		opt_options_cuda,
		min_xyz,
		grid_step,
		grid_num,
		empty_grid_cuda,
		current_static_grids_cuda,
		has_soup
	);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to compute 3d features:  %3.1f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}



void FORWARD::compute3dfeatures(
	const int N, int D, int M,
	const int S, const int P,
	const float* samples_pos,
	const int* sample_neighbours,
	const int* sample_idx_itselves,
	const float* pos_cuda,
	float* rot_cuda,
	const float* scale_cuda,
	const float* opacity_cuda,
	const float* shs_cuda,
	const float* half_length_cuda,
	float* sigma_inv_cuda,
	float* result,
	float* feature_opacity_cuda
	)
{
	// cudaDeviceSynchronize();
	// Init3dfeaturesCUDA <<< (S*M*D + 255) / 256, 256 >>>(
	// 	S*M*D,
	// 	result
	// );
	// cudaDeviceSynchronize();
	// Init3dfeaturesCUDA <<< (S + 255) / 256, 256 >>>(
	// 	S,
	// 	feature_opacity_cuda
	// );

	// cudaDeviceSynchronize();
	// compute3dcovarianceCUDA<< <(P + 255) / 256, 256 >> >(
	// 	P,
	// 	rot_cuda,
	// 	scale_cuda,
	// 	sigma_inv_cuda,
	// 	0.0,
	// 	opt_options_cuda
	// );
	// cudaDeviceSynchronize();
	// compute3dfeaturesCUDA << <(N + 255) / 256, 256 >> > (
	// 	N, D, M,
	// 	S,
	// 	samples_pos,
	// 	sample_neighbours,
	// 	sample_idx_itselves,
	// 	pos_cuda,
	// 	rot_cuda,
	// 	scale_cuda,
	// 	opacity_cuda,
	// 	shs_cuda,
	// 	half_length_cuda,
	// 	sigma_inv_cuda,
	// 	result,
	// 	feature_opacity_cuda
	// );
	// cudaDeviceSynchronize();
}

void FORWARD::computeL1loss3d(
	const int valid_grid_num, int D, int M,
	const int P, int S_PerGird,
	const float* aim_feature_cuda,
	const float* cur_feature_cuda,
	const float* aim_opacity_cuda,
	const float* cur_opacity_cuda,
	float* opacity_grad_cuda,
	float* feature_grad_cuda,
	float* total_feature_loss,
	float* total_shape_loss,
	bool* grid_is_converged_cuda,
	bool* grid_nearly_converged_cuda,
	float* grid_loss_sums_cuda,
	bool* opt_options_cuda,
	int* empty_grid_cuda,
	bool adjust_op_range,
	bool has_soup
){
	int S = valid_grid_num * S_PerGird;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop) ;
	cudaEventRecord(start, 0);

	Init3dfeaturesCUDA <<< (S*D*M + 255) / 256, 256 >>>(
		S*D*M,
		feature_grad_cuda
	);
	cudaDeviceSynchronize();
	Init3dfeaturesCUDA <<< (valid_grid_num + 255) / 256, 256 >>>(
		valid_grid_num,
		grid_loss_sums_cuda
	);
	cudaDeviceSynchronize();
	Init3dfeaturesCUDA <<< 1, 256 >>>(
		1,
		total_feature_loss
	);
	cudaDeviceSynchronize();
	Init3dfeaturesCUDA <<< 1, 256 >>>(
		1,
		total_shape_loss
	);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// printf("Time to initial loss:  %3.1f ms\n", elapsedTime);

	compute3dlossCUDA <<<valid_grid_num, S_PerGird>>>(
		D, M, S,
		aim_feature_cuda,
		cur_feature_cuda,
		aim_opacity_cuda,
		cur_opacity_cuda,
		opacity_grad_cuda,
		feature_grad_cuda,
		total_feature_loss,
		total_shape_loss,
		grid_is_converged_cuda,
		grid_nearly_converged_cuda,
		grid_loss_sums_cuda,
		opt_options_cuda
	);
	cudaDeviceSynchronize();

	cudaEventRecord(start, 0) ;
	cudaEventSynchronize(start);
	cudaEventElapsedTime(&elapsedTime, stop, start);
	printf("Time to compute 3d loss:  %3.1f ms\n", elapsedTime);

	if (adjust_op_range){
		judge_convergencyCUDA<<< (valid_grid_num + 255) / 256, 256 >>>(
			S_PerGird,
			valid_grid_num,
			grid_is_converged_cuda,
			grid_nearly_converged_cuda,
			grid_loss_sums_cuda,
			has_soup
		);
		cudaDeviceSynchronize();
	}

	// cudaEventRecord(stop, 0) ;
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&elapsedTime, start, stop);
	// printf("Time to judge grids' convergency:  %3.1f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	int2* rects,
	float3 boxmin,
	float3 boxmax,
	bool _to_ortho,
	float ortho_scale)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		rects,
		boxmin,
		boxmax,
		_to_ortho,
		ortho_scale
		);
}