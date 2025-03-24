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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* is_surface,
			int* radii = nullptr,
			int* rects = nullptr,
			float* boxmin = nullptr,
			float* boxmax = nullptr,
			bool _to_ortho = false,
			float ortho_scale = 2.0f);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot);

		static void forward3d_grid(
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
		);


		static void forward3d(
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
			float* sigma_cuda,
			float* result,
			float* feature_opacity_cuda
		);

		static void backward3d_grid(
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
		);

		static void backward3d(
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
		);

		static void L1loss3d(
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
		);

		static void update3d(
			const int P, int D, int M,
			const float* dF_dopacity,
			const float* dF_dshs,
			const float* dF_dpos_cuda,
			const float* dF_drot_cuda,
			const float* dF_dscale_cuda,
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
		);
	};
};

#endif