#pragma once

#include "cuda_utils.h"
#include "float3.h"
#include "lcg_rng.h"

/* Disney BSDF functions, for additional details and examples see:
 * - https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
 * - https://www.shadertoy.com/view/XdyyDd
 * - https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
 * - https://schuttejoe.github.io/post/disneybsdf/
 *
 * Variable naming conventions with the Burley course notes:
 * V -> w_o
 * L -> w_i
 * H -> w_h
 */

struct DisneyMaterial {
	float3 base_color;
	float metallic;

	float specular;
	float roughness;
	float specular_tint;
	float anisotropy;

	float sheen;
	float sheen_tint;
	float clearcoat;
	float clearcoat_gloss;

	float ior;
	float specular_transmission;
	float transmission_roughness;
	float flatness;
};

__device__ bool same_hemisphere(const float3 &w_o, const float3 &w_i, const float3 &n) {
	return dot(w_o, n) * dot(w_i, n) > 0.f;
}

__device__ bool relative_ior(const float3 &w_o, const float3 &n, float ior, float &eta_o, float &eta_i)
{
	bool entering = dot(w_o, n) > 0.f;
	eta_i = entering ? 1.f : ior;
	eta_o = entering ? ior : 1.f;
	return entering;
}

// Sample the hemisphere using a cosine weighted distribution,
// returns a vector in a hemisphere oriented about (0, 0, 1)
__device__ float3 cos_sample_hemisphere(float2 u) {
	float2 s = 2.f * u - make_float2(1.f);
	float2 d;
	float radius = 0.f;
	float theta = 0.f;
	if (s.x == 0.f && s.y == 0.f) {
		d = s;
	} else {
		if (fabs(s.x) > fabs(s.y)) {
			radius = s.x;
			theta  = M_PI / 4.f * (s.y / s.x);
		} else {
			radius = s.y;
			theta  = M_PI / 2.f - M_PI / 4.f * (s.x / s.y);
		}
	}
	d = radius * make_float2(cos(theta), sin(theta));
	return make_float3(d.x, d.y, sqrt(max(0.f, 1.f - d.x * d.x - d.y * d.y)));
}

__device__ float3 spherical_dir(float sin_theta, float cos_theta, float phi) {
	return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

__device__ float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g) {
	float f = n_f * pdf_f;
	float g = n_g * pdf_g;
	return (f * f) / (f * f + g * g);
}

__device__ float schlick_weight(float cos_theta) {
	return pow(saturate(1.f - cos_theta), 5.f);
}

// Complete Fresnel Dielectric computation, for transmission at ior near 1
// they mention having issues with the Schlick approximation.
// eta_i: material on incident side's ior
// eta_t: material on transmitted side's ior
__device__ float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t) {
	float g = pow2(eta_t) / pow2(eta_i) - 1.f + pow2(cos_theta_i);
	if (g < 0.f) {
		return 1.f;
	}
	return 0.5f * pow2(g - cos_theta_i) / pow2(g + cos_theta_i)
		* (1.f + pow2(cos_theta_i * (g + cos_theta_i) - 1.f) / pow2(cos_theta_i * (g - cos_theta_i) + 1.f));
}

// D_GTR1: Generalized Trowbridge-Reitz with gamma=1
// Burley notes eq. 4
__device__ float gtr_1(float cos_theta_h, float alpha) {
	if (alpha >= 1.f) {
		return M_1_PI;
	}
	float alpha_sqr = alpha * alpha;
	return M_1_PI * (alpha_sqr - 1.f) / (log(alpha_sqr) * (1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h));
}

// D_GTR2: Generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 8
__device__ float gtr_2(float cos_theta_h, float alpha) {
	float alpha_sqr = alpha * alpha;
	return M_1_PI * alpha_sqr / max(pow2(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h), SMALL_EPSILON);
}

// D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 13
__device__ float gtr_2_aniso(float h_dot_n, float h_dot_x, float h_dot_y, float2 alpha) {
	return M_1_PI / max((alpha.x * alpha.y * pow2(pow2(h_dot_x / alpha.x) + pow2(h_dot_y / alpha.y) + h_dot_n * h_dot_n)), SMALL_EPSILON);
}

__device__ float smith_shadowing_ggx(float n_dot_o, float alpha_g) {
	float a = alpha_g * alpha_g;
	float b = n_dot_o * n_dot_o;
	return 1.f / (n_dot_o + sqrt(a + b - a * b));
}

__device__ float smith_shadowing_ggx_aniso(float n_dot_o, float o_dot_x, float o_dot_y, float2 alpha) {
	return 1.f / (n_dot_o + sqrt(pow2(o_dot_x * alpha.x) + pow2(o_dot_y * alpha.y) + pow2(n_dot_o)));
}

// Sample a reflection direction the hemisphere oriented along n and spanned by v_x, v_y using the random samples in s
__device__ float3 sample_lambertian_dir(const float3 &n, const float3 &v_x, const float3 &v_y, const float2 &s) {
	const float3 hemi_dir = normalize(cos_sample_hemisphere(s));
	return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

// Sample the microfacet normal vectors for the various microfacet distributions
__device__ float3 sample_gtr_1_h(const float3 &n, const float3 &v_x, const float3 &v_y, float alpha, const float2 &s) {
	float phi_h = 2.f * M_PI * s.x;
	float alpha_sqr = alpha * alpha;
	float cos_theta_h_sqr = (1.f - pow(alpha_sqr, 1.f - s.y)) / (1.f - alpha_sqr);
	float cos_theta_h = sqrt(cos_theta_h_sqr);
	float sin_theta_h = 1.f - cos_theta_h_sqr;
	float3 hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
	return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

__device__ float3 sample_gtr_2_h(const float3 &n, const float3 &v_x, const float3 &v_y, float alpha, const float2 &s) {
	float phi_h = 2.f * M_PI * s.x;
	float cos_theta_h_sqr = (1.f - s.y) / (1.f + (alpha * alpha - 1.f) * s.y);
	float cos_theta_h = sqrt(cos_theta_h_sqr);
	float sin_theta_h = 1.f - cos_theta_h_sqr;
	float3 hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
	return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

__device__ float3 sample_gtr_2_aniso_h(const float3 &n, const float3 &v_x, const float3 &v_y, const float2 &alpha, const float2 &s) {
	float x = 2.f * M_PI * s.x;
	float3 w_h = sqrt(s.y / (1.f - s.y)) * (alpha.x * cos(x) * v_x + alpha.y * sin(x) * v_y) + n;
	return normalize(w_h);
}

__device__ float lambertian_pdf(const float3 &w_i, const float3 &n) {
	float d = dot(w_i, n);
	if (d > 0.f) {
		return d * M_1_PI;
	}
	return 0.f;
}

__device__ float gtr_1_pdf(const float3 &w_o, const float3 &w_i, const float3 &n, float alpha) {
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float3 w_h = normalize(w_i + w_o);
	float cos_theta_h = dot(n, w_h);
	float d = gtr_1(cos_theta_h, alpha);
	return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

__device__ float gtr_2_pdf(const float3 &w_o, const float3 &w_i, const float3 &n, float alpha) {
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float3 w_h = normalize(w_i + w_o);
	float cos_theta_h = dot(n, w_h);
	float d = gtr_2(cos_theta_h, alpha);
	return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

__device__ float gtr_2_transmission_pdf(const float3 &w_o, const float3 &w_i, const float3 &n,
	float alpha, float ior)
{
	if (same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float eta_o, eta_i;
	relative_ior(w_o, n, ior, eta_i, eta_o);

	// From Eq 16 of Microfacet models for refraction
	float3 w_h = -(w_o * eta_o + w_i * eta_i);
	w_h = normalize(w_h);

	// // float3 w_h = normalize(w_i + w_o);
	// float cos_theta_h = fabs(dot(n, w_h));
	// float d = gtr_2(cos_theta_h, alpha);
	// return d * cos_theta_h / (4.f * fabs(dot(w_o, w_h)));


	// float3 w_h = normalize(w_o + w_i * eta_i / eta_o);
	float cos_theta_h = fabs(dot(n, w_h));
	float i_dot_h = dot(w_i, w_h);
	float o_dot_h = dot(w_o, w_h);
	float d = gtr_2(cos_theta_h, alpha);
	float dwh_dwi = o_dot_h * pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);
	return d * cos_theta_h * fabs(dwh_dwi);
}

__device__ float gtr_2_aniso_pdf(const float3 &w_o, const float3 &w_i, const float3 &n,
	const float3 &v_x, const float3 &v_y, const float2 alpha)
{
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float3 w_h = normalize(w_i + w_o);
	float cos_theta_h = dot(n, w_h);
	float d = gtr_2_aniso(cos_theta_h, fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
	return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

__device__ float3 disney_diffuse(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i)
{
	float3 w_h = normalize(w_i + w_o);
	float n_dot_o = fabs(dot(w_o, n));
	float n_dot_i = fabs(dot(w_i, n));
	float i_dot_h = dot(w_i, w_h);
	float fd90 = 0.5f + 2.f * mat.roughness * i_dot_h * i_dot_h;
	float fi = schlick_weight(n_dot_i);
	float fo = schlick_weight(n_dot_o);
	return mat.base_color * M_1_PI * lerp(1.f, fd90, fi) * lerp(1.f, fd90, fo);
}

__device__ float3 disney_subsurface(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i) {
	float3 w_h = normalize(w_i + w_o);
    float n_dot_o = fabs(dot(w_o, n));
	float n_dot_i = fabs(dot(w_i, n));
	float i_dot_h = dot(w_i, w_h);

    float FL = schlick_weight(n_dot_i), FV = schlick_weight(n_dot_o);
    float Fss90 = i_dot_h*i_dot_h * mat.roughness;
    float Fss = lerp(1.0, Fss90, FL) * lerp(1.0, Fss90, FV);
    float ss = 1.25 * (Fss * (1. / (n_dot_i + n_dot_o) - .5) + .5);
    
    return (1./M_PI) * ss * mat.base_color;
}

// Eavg in the algorithm is fitted into this
__device__ float AverageEnergy(float rough){
    float smoothness = 1.0 - rough;
    float r = -0.0761947 - 0.383026 * smoothness;
          r = 1.04997 + smoothness * r;
          r = 0.409255 + smoothness * r;
    return min(0.9, r); 
}

// multiple scattering...
// Favg in the algorithm is fitted into this
__device__ float3 AverageFresnel(float3 specularColor){
    return specularColor + (make_float3(1.0) - specularColor) * (1.0 / 21.0);
}

__device__ float3 disney_multiscatter(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i,
	cudaTextureObject_t GGX_E_LOOKUP, cudaTextureObject_t GGX_E_AVG_LOOKUP)
{
	float3 w_h = normalize(w_i + w_o);
	float v_dot_n = abs(dot(normalize(w_o), normalize(n)));
	float l_dot_n = abs(dot(normalize(w_i), normalize(n)));
	float alpha = max(mat.roughness*mat.roughness, MIN_ALPHA);

    //E(μ) is in fact the sum of the red and green channels in our environment BRDF
    // vec2 sampleE_o = texture2D(BRDFlut, vec2(NdotV, alpha)).xy;
    // E_o = sampleE_o.x + sampleE_o.y;
	float E_o = tex2D<float>(GGX_E_LOOKUP, v_dot_n, alpha);
    float oneMinusE_o = 1.0 - E_o;
    float E_i = tex2D<float>(GGX_E_LOOKUP, l_dot_n, alpha);
    float oneMinusE_i = 1.0 - E_i;

    float Eavg = tex2D<float>(GGX_E_AVG_LOOKUP, alpha, .5);
	// float Eavg = AverageEnergy(alpha);
    float oneMinusEavg = 1.0 - Eavg;
	
	float lum = luminance(mat.base_color);
	float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
	float3 F0 = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);
	float3 Favg = AverageFresnel(F0);

    float brdf = (oneMinusE_o * oneMinusE_i) / (M_PI * oneMinusEavg);
    float3 energyScale = (Favg * Favg * Eavg) / (make_float3(1.0) - Favg * oneMinusEavg);

    return brdf * energyScale;
}

__device__ float G(float3 i, float3 o, float3 h, float alpha)
{
	alpha = 1.f - alpha;
	// Roughly follows Eq 23 from Microfacet Models for Refraction.
	// G is approximately the seperable product of two monodirectional shadowing terms G1 (aka smith shadowing function)
	return smith_shadowing_ggx(fabs(dot(i, h)), alpha) * smith_shadowing_ggx(fabs(dot(o, h)), alpha);
}

__device__ float D(float3 m, float3 n, float alpha)
{
	// alpha = 1.f - alpha;
	// float alpha_sqr = alpha * alpha;

	// From Eq 33 of Microfacet Models for Refraction
	// help from http://filmicworlds.com/blog/optimizing-ggx-shaders-with-dotlh/
	float m_dot_n = dot(m, n);
	float posCharFunc = (m_dot_n > 0) ? 1.f : 0.f;
	float alpha_sqr = pow2(alpha);
	float denom = m_dot_n * m_dot_n * (alpha_sqr - 1.f) + 1.f;
	float D = alpha_sqr / (M_PI * denom * denom);
	return D;
	// M_1_PI * alpha_sqr / max(pow2(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h), SMALL_EPSILON);
	// return gtr_2(fabs(dot(n, w_ht)), alpha);
}

__device__ float F(float3 i, float3 m, float eta_t, float eta_i)
{
	// From Eq 22 of Microfacet Models for Refraction
	float c = fabs(dot(i, m));
	float g = pow2(eta_t) / pow2(eta_i) - 1 + pow2(c);
	if (g < 0) return 1; // if g is imaginary after sqrt, this indicates a total internal reflection
	g = sqrtf(g);
	float f = .5f;
	f *= pow2(g - c) / pow2(g + c);
	f *= (1.f + (pow2(c * (g + c) - 1) / pow2(c * (g - c) + 1)));
	return f;
}

__device__ float3 disney_microfacet_isotropic(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i)
{
	float3 w_h = normalize(w_i + w_o);
	float lum = luminance(mat.base_color);
	float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
	float3 spec = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

	float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
	float d = gtr_2(dot(n, w_h), alpha);
	float3 f = lerp(spec, make_float3(1.f), schlick_weight(dot(w_i, w_h)));
	float g = smith_shadowing_ggx(dot(n, w_i), alpha) * smith_shadowing_ggx(dot(n, w_o), alpha);
	return d * f * g;
}

__device__ float3 disney_microfacet_transmission_isotropic(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i)
{	

	float eta_o, eta_i;
	bool entering = relative_ior(w_o, n, mat.ior, eta_o, eta_i);

	float alpha = max(0.001f, mat.transmission_roughness * mat.transmission_roughness);
	
	// From Eq 16 of Microfacet models for refraction
	float3 w_ht = -(normalize(w_o) * eta_i + normalize(w_i) * eta_o);
	w_ht = normalize(w_ht);

	// w_ht = n; // HACK

	float3 w_r = refract(-w_o, (entering) ? w_ht : -w_ht, eta_o / eta_i);

	// float3 n_ = (dot(w_ht, n) > 0) ? n : -;

	float d = D(w_ht, n, alpha); 
	float f = F(w_i, w_ht, eta_o, eta_i);
	float g = G(w_i, w_o, w_ht, alpha); 

	float i_dot_h = fabs(dot(w_i, w_ht));
	float o_dot_h = fabs(dot(w_o, w_ht));
	float i_dot_n = fabs(dot(w_i, n));
	float o_dot_n = fabs(dot(w_o, n));

	if (o_dot_n == 0.f || i_dot_n == 0.f) {
		return make_float3(0.f);
	}

	// From Eq 21 of Microfacet models for refraction
	float c = (fabs(i_dot_h) * fabs(o_dot_h)) / (fabs(i_dot_n) * fabs(o_dot_n));
	c *= pow2(eta_o) / pow2(eta_i * i_dot_h + eta_o * o_dot_h);

	//// hacking in a spherical gaussian here... Can't seem to get microfacet model working...
	return mat.base_color * abs( pow(dot(w_ht, n), (1.0f / (alpha + EPSILON))) ); //* c;//g; //c * (1.f - f) * g * d;
}

__device__ float3 disney_microfacet_anisotropic(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i, const float3 &v_x, const float3 &v_y)
{
	float3 w_h = normalize(w_i + w_o);
	float lum = luminance(mat.base_color);
	float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
	float3 spec = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

	float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
	float a = mat.roughness * mat.roughness;
	float2 alpha = make_float2(max(MIN_ALPHA, a / aspect), max(MIN_ALPHA, a * aspect));
	float d = gtr_2_aniso(dot(n, w_h), fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
	float3 f = lerp(spec, make_float3(1.f), schlick_weight(dot(w_i, w_h)));
	float g = smith_shadowing_ggx_aniso(dot(n, w_i), fabs(dot(w_i, v_x)), fabs(dot(w_i, v_y)), alpha)
		* smith_shadowing_ggx_aniso(dot(n, w_o), fabs(dot(w_o, v_x)), fabs(dot(w_o, v_y)), alpha);
	return d * f * g;
}

__device__ float disney_clear_coat(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i)
{
	float3 w_h = normalize(w_i + w_o);
	float alpha = lerp(0.1f, MIN_ALPHA, mat.clearcoat_gloss);
	float d = gtr_1(dot(n, w_h), alpha);
	float f = lerp(0.04f, 1.f, schlick_weight(dot(w_i, n)));
	float g = smith_shadowing_ggx(dot(n, w_i), 0.25f) * smith_shadowing_ggx(dot(n, w_o), 0.25f);
	return 0.25f * mat.clearcoat * d * f * g;
}

__device__ float3 disney_sheen(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i)
{
	float3 w_h = normalize(w_i + w_o);
	float lum = luminance(mat.base_color);
	float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
	float3 sheen_color = lerp(make_float3(1.f), tint, mat.sheen_tint);
	float f = schlick_weight(dot(w_i, n));
	return f * mat.sheen * sheen_color;
}

__device__ float3 disney_brdf(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i, const float3 &v_x, const float3 &v_y,
	cudaTextureObject_t GGX_E_LOOKUP, cudaTextureObject_t GGX_E_AVG_LOOKUP
	)
{

	if (!same_hemisphere(w_o, w_i, n)) {
		// transmissive objects refract when back of surface is visible.
		if (mat.specular_transmission > 0.f) 
		{
			float3 spec_trans = disney_microfacet_transmission_isotropic(mat, n, w_o, w_i);
			return spec_trans * (1.f - mat.metallic) * mat.specular_transmission;
		}

		// non-transmissive objects appear black when back of surface is visible.
		return make_float3(0.f);
	}

	float coat = disney_clear_coat(mat, n, w_o, w_i);
	float3 sheen = disney_sheen(mat, n, w_o, w_i);
	float3 diffuse = disney_diffuse(mat, n, w_o, w_i);
	float3 subsurface = disney_subsurface(mat, n, w_o, w_i);
	float3 gloss;
	if (mat.anisotropy == 0.f) {
		gloss = disney_microfacet_isotropic(mat, n, w_o, w_i);
		gloss = gloss + disney_multiscatter(mat, n, w_o, w_i, GGX_E_LOOKUP, GGX_E_AVG_LOOKUP);
	} else 
	{
		gloss = disney_microfacet_anisotropic(mat, n, w_o, w_i, v_x, v_y);
		gloss = gloss + disney_multiscatter(mat, n, w_o, w_i, GGX_E_LOOKUP, GGX_E_AVG_LOOKUP);
	}
	return 
		(lerp(diffuse, subsurface, mat.flatness) * (1.f - mat.metallic) * (1.f - mat.specular_transmission) 
		+ sheen + gloss + coat) * fabs(dot(w_i, n));
}

__device__ float disney_pdf(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &w_i, const float3 &v_x, const float3 &v_y
) {
	bool entering = dot(w_o, n) > 0.f;
	bool sameHemisphere = same_hemisphere(w_o, w_i, n);
	
	float alpha = max(0.002f, mat.roughness * mat.roughness);
	float t_alpha = max(0.002f, mat.transmission_roughness * mat.transmission_roughness);
	float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
	float2 alpha_aniso = make_float2(max(0.002f, alpha / aspect), max(0.002f, alpha * aspect));

	float clearcoat_alpha = lerp(0.1f, MIN_ALPHA, mat.clearcoat_gloss);

	float diffuse = lambertian_pdf(w_i, n);
	float clear_coat = gtr_1_pdf(w_o, w_i, n, clearcoat_alpha);

	float n_comp = 3.f;
	float microfacet = 0.f;
	float microfacet_transmission = 0.f;
	if (mat.anisotropy == 0.f) {
		microfacet = gtr_2_pdf(w_o, w_i, n, alpha);
	} else {
		microfacet = gtr_2_aniso_pdf(w_o, w_i, n, v_x, v_y, alpha_aniso);
	}

	if ((mat.specular_transmission > 0.f) && (!same_hemisphere(w_o, w_i, n))) {
		// microfacet_transmission = gtr_2_transmission_pdf(w_o, w_i, n, alpha, mat.ior);
		
		// HACK
		microfacet_transmission = 1.f;
	} 

	// not sure why, but energy seems to be added from smooth metallic. By subtracting mat.metallic from n_comps,
	// we decrease brightness and become almost perfectly conserving energy for shiny metallic. As metals get 
	// rough, we lose energy from our single scattering microfacet model around .1 roughness, so we 
	// remove the energy reduction kludge for metallic in the case of slightly rough metal. We still 
	// seem to lose a lot of energy in that case, and could likely benefit from a multiple scattering microfacet
	// model. 
	// For transmission, so long as we subtract 1 from the components, we seem to preserve energy
	// regardless if the transmission is rough or smooth.
	float metallic_kludge = max(mat.metallic - mat.roughness * 10.f, 0.f);
	float transmission_kludge = mat.specular_transmission;
	n_comp -= lerp(transmission_kludge, metallic_kludge, mat.metallic); 
	return (diffuse + microfacet + microfacet_transmission + clear_coat) / n_comp;
}

/* Sample a component of the Disney BRDF, returns the sampled BRDF color,
 * ray reflection direction (w_i) and sample PDF.
 */
__device__ float3 sample_disney_brdf(const DisneyMaterial &mat, const float3 &n,
	const float3 &w_o, const float3 &v_x, const float3 &v_y, LCGRand &rng,
	float3 &w_i, float &pdf, bool &is_specular,
	cudaTextureObject_t GGX_E_LOOKUP, cudaTextureObject_t GGX_E_AVG_LOOKUP)
{
	bool entering = dot(w_o, n) > 0.f;

	int component = 0;
	if (mat.specular_transmission == 0.f) {
		component = lcg_randomf(rng) * 3.f;
		component = glm::clamp(component, 0, 2);
	} else 
	{
		if (entering) {
			component = lcg_randomf(rng) * 4.f;
			component = glm::clamp(component, 0, 3);
		}
		// HACK, forcing only refractive brdf when entering surface 
		else component = 3; 
	}

	is_specular = ((component != 0) && (component != 3));

	float2 samples = make_float2(lcg_randomf(rng), lcg_randomf(rng));
	if (component == 0) {
		// Sample diffuse component
		w_i = sample_lambertian_dir(n, v_x, v_y, samples);
	} else if (component == 1) {
		float3 w_h;
		float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
		if (mat.anisotropy == 0.f) {
			w_h = sample_gtr_2_h(n, v_x, v_y, alpha, samples);
		} else {
			float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
			float2 alpha_aniso = make_float2(max(MIN_ALPHA, alpha / aspect), max(MIN_ALPHA, alpha * aspect));
			w_h = sample_gtr_2_aniso_h(n, v_x, v_y, alpha_aniso, samples);
		}
		w_i = reflect(-w_o, w_h);

		// Invalid reflection, terminate ray
		if (!same_hemisphere(w_o, w_i, n)) {
			pdf = 0.f;
			w_i = make_float3(0.f);
			return make_float3(0.f);
		}
	} else if (component == 2) {
		// Sample clear coat component
		float alpha = lerp(0.1f, MIN_ALPHA, mat.clearcoat_gloss);
		float3 w_h = sample_gtr_1_h(n, v_x, v_y, alpha, samples);
		w_i = reflect(-w_o, w_h);

		// Invalid reflection, terminate ray
		if (!same_hemisphere(w_o, w_i, n)) {
			pdf = 0.f;
			w_i = make_float3(0.f);
			return make_float3(0.f);
		}
	} else {	
		// Sample microfacet transmission component
		float alpha = max(MIN_ALPHA, mat.transmission_roughness * mat.transmission_roughness);
		float3 w_h = sample_gtr_2_h(n, v_x, v_y, alpha, samples);
		float eta_o, eta_i;
		bool entering = relative_ior(w_o, w_h, mat.ior, eta_o, eta_i);
		// w_i = refract(-w_o, w_h, eta_o / eta_i);

		// w_o is flipped, so we also flip eta
		w_i = refract(-w_o, (entering) ? w_h : -w_h, eta_i / eta_o);

		// Total internal reflection
		if (all_zero(w_i)) {
			w_i = reflect(-w_o, (entering) ? w_h : -w_h);
			pdf = 1.f;
			return make_float3(1.f);
		}
	// return make_float3(1.f); // HACK
	}
	pdf = disney_pdf(mat, n, w_o, w_i, v_x, v_y);

	float3 brdf = disney_brdf(mat, n, w_o, w_i, v_x, v_y, GGX_E_LOOKUP, GGX_E_AVG_LOOKUP);
	return brdf;
}

