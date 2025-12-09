#ifdef __cplusplus
#include <pxr/base/gf/vec3f.h>

struct ISpectrum {
    // virtual pxr::GfVec3f toFloat3() = 0;
};

struct RGBSpectrum : ISpectrum {
    RGBSpectrum(const pxr::GfVec3f& data) : data(data)
    {
    }
    pxr::GfVec3f data;
    float padding;
};
#else

// ============================================================================
// Spectrum Configuration
// ============================================================================
// Toggle between RGB and Sampled spectrum by changing this define
// #define USE_RGB_SPECTRUM 1  // Use RGB spectrum (default)
// #define USE_SAMPLED_SPECTRUM 1  // Uncomment to use sampled spectrum

interface ISpectrum {
    // Convert the spectrum to a float3 representation
    float3 toFloat3();
}

// ============================================================================
// CIE 1931 XYZ Color Matching Functions (Multi-lobe Gaussian approximation)
// Reference: "Simple Analytic Approximations to the CIE XYZ Color Matching
// Functions"
// For Slang GPU side, we use the approximation since exact lookup tables are
// too large
// ============================================================================
float gaussianXYZ(float x, float lambda, float invSigma1, float invSigma2)
{
    float t = (x - lambda) * (x < lambda ? invSigma1 : invSigma2);
    return exp(-0.5f * t * t);
}

float3 wavelengthToXYZ(float lambda)
{
    // Multi-lobe piecewise Gaussian fit for CIE XYZ color matching functions
    // Based on "Simple Analytic Approximations to the CIE XYZ Color Matching
    // Functions" - Wyman et al., JCGT 2013
    // This gives good accuracy for GPU-side spectral calculations

    float x = 0.362f * gaussianXYZ(lambda, 442.0f, 0.0624f, 0.0374f) +
              1.056f * gaussianXYZ(lambda, 599.8f, 0.0264f, 0.0323f) -
              0.065f * gaussianXYZ(lambda, 501.1f, 0.0490f, 0.0382f);

    float y = 0.821f * gaussianXYZ(lambda, 568.8f, 0.0213f, 0.0247f) +
              0.286f * gaussianXYZ(lambda, 530.9f, 0.0613f, 0.0322f);

    float z = 1.217f * gaussianXYZ(lambda, 437.0f, 0.0845f, 0.0278f) +
              0.681f * gaussianXYZ(lambda, 459.0f, 0.0385f, 0.0725f);

    return float3(x, y, z);
}

// ============================================================================
// D65 Standard Illuminant
// Reference: CIE Standard Illuminant D65 (daylight)
// ============================================================================
float wavelengthToD65(float lambda)
{
    // D65 illuminant spectral power distribution (normalized)
    // More precise values sampled at regular intervals
    // Based on: https://en.wikipedia.org/wiki/Illuminant_D65

    if (lambda < 300.0f || lambda > 830.0f)
        return 0.0f;

    // D65 values at 5nm intervals from 300-830nm (107 samples)
    // These are normalized values
    const float d65_table[] = {
        0.034f,   1.664f,   3.294f,   11.765f,  20.236f,  28.644f,  37.053f,
        38.501f,  39.948f,  42.430f,  44.911f,  45.775f,  46.638f,  49.363f,
        52.089f,  51.032f,  49.975f,  52.311f,  54.648f,  68.701f,  82.754f,
        87.120f,  91.486f,  92.458f,  93.431f,  90.057f,  86.682f,  95.773f,
        104.864f, 110.936f, 117.008f, 117.410f, 117.812f, 116.336f, 114.861f,
        115.392f, 115.923f, 112.367f, 108.811f, 109.082f, 109.354f, 108.578f,
        107.802f, 106.296f, 104.790f, 106.239f, 107.689f, 106.047f, 104.405f,
        104.225f, 104.046f, 102.023f, 100.000f, 98.167f,  96.334f,  96.061f,
        95.788f,  92.236f,  88.685f,  89.345f,  90.006f,  89.802f,  89.599f,
        88.648f,  87.698f,  85.493f,  83.288f,  83.493f,  83.699f,  81.863f,
        80.026f,  80.120f,  80.214f,  81.246f,  82.277f,  80.281f,  78.284f,
        74.002f,  69.721f,  70.665f,  71.609f,  72.979f,  74.349f,  67.976f,
        61.604f,  65.744f,  69.885f,  72.486f,  75.087f,  69.339f,  63.592f,
        55.005f,  46.418f,  56.611f,  66.805f,  65.094f,  63.382f,  63.843f,
        64.304f,  61.877f,  59.451f,  55.705f,  51.959f,  54.699f,  57.440f,
        58.876f,  60.312f
    };

    // Normalize lambda to table index (5nm spacing from 300nm)
    float idx = (lambda - 300.0f) / 5.0f;

    if (idx < 0.0f)
        return d65_table[0];
    if (idx >= 106.0f)
        return d65_table[106];

    // Linear interpolation between table values
    int i0 = int(idx);
    int i1 = min(i0 + 1, 106);
    float t = frac(idx);

    return lerp(d65_table[i0], d65_table[i1], t);
}

// ============================================================================
// RGB to XYZ Conversion (Rec. 709 with D65 illuminant)
// ============================================================================
float3 rgbToXYZ_Rec709(float3 rgb)
{
    // RGB (Rec. 709) to XYZ conversion matrix (D65 illuminant)
    float3 xyz;
    xyz.x = 0.4124f * rgb.r + 0.3576f * rgb.g + 0.1805f * rgb.b;
    xyz.y = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
    xyz.z = 0.0193f * rgb.r + 0.1192f * rgb.g + 0.9505f * rgb.b;
    return xyz;
}

// ============================================================================
// XYZ to RGB (Rec. 709 with D65 illuminant)
// ============================================================================
float3 xyzToRGB_Rec709(float3 xyz)
{
    // XYZ to Rec. 709 RGB conversion matrix (D65 illuminant)
    float3 rgb;
    rgb.r = 3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z;
    rgb.g = -0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z;
    rgb.b = 0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z;

    return rgb;
}

struct RGBSpectrum : ISpectrum {
    __init(float v)
    {
        data = float3(v, v, v);
        padding = 0;
    }

    __init(float3 rgb)
    {
        data = rgb;
        padding = 0;
    }

    float3 data;
    float padding;

    float3 toFloat3()
    {
        return data;
    }
};

// Helper functions for RGBSpectrum operations
RGBSpectrum add(RGBSpectrum a, RGBSpectrum b)
{
    return RGBSpectrum(a.data + b.data);
}

RGBSpectrum sub(RGBSpectrum a, RGBSpectrum b)
{
    return RGBSpectrum(a.data - b.data);
}

RGBSpectrum mul(RGBSpectrum a, RGBSpectrum b)
{
    return RGBSpectrum(a.data * b.data);
}

RGBSpectrum mul(RGBSpectrum a, float scalar)
{
    return RGBSpectrum(a.data * scalar);
}

RGBSpectrum mul(float scalar, RGBSpectrum a)
{
    return RGBSpectrum(a.data * scalar);
}

RGBSpectrum div(RGBSpectrum a, float scalar)
{
    return RGBSpectrum(a.data / scalar);
}

// ============================================================================
// Sampled Spectrum Implementation (8-sample version for testing)
// Uses wavelength range 380nm-720nm (visible spectrum)
// ============================================================================
struct SampledSpectrum : ISpectrum {
    static const int NUM_SAMPLES = 8;
    static const float LAMBDA_MIN = 380.0;  // nm
    static const float LAMBDA_MAX = 720.0;  // nm

    __init(float v)
    {
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            this.samples[i] = v;
        }
    }

    __init(float3 rgb)
    {
        // Convert RGB to spectrum using improved reconstruction
        fromRGB(rgb);
    }

    float samples[NUM_SAMPLES];

    [mutating]
    void fromRGB(float3 rgb)
    {
        // Convert RGB to XYZ first
        float3 xyz = RGBtoXYZ(rgb);
        
        // For each wavelength sample, compute the contribution
        // We use a simple reconstruction based on the XYZ color matching functions
        float lambdaStep = (LAMBDA_MAX - LAMBDA_MIN) / float(NUM_SAMPLES - 1);
        
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            float lambda = LAMBDA_MIN + float(i) * lambdaStep;
            
            // Get XYZ color matching functions at this wavelength
            float3 cmf = wavelengthToXYZ(lambda);
            
            // Reconstruct spectral sample as a weighted sum
            // This is a simplified smits-style RGB to spectrum conversion
            this.samples[i] = dot(xyz, cmf);
        }
    }

    static float3 RGBtoXYZ(float3 rgb)
    {
        return rgbToXYZ_Rec709(rgb);
    }

    static float3 XYZtoRGB(float3 xyz)
    {
        return xyzToRGB_Rec709(xyz);
    }

    static float3 spectrumToXYZ(SampledSpectrum spec)
    {
        // Integrate spectrum against CIE XYZ color matching functions
        float3 xyz = float3(0.0, 0.0, 0.0);
        float lambdaStep = (LAMBDA_MAX - LAMBDA_MIN) / float(NUM_SAMPLES - 1);
        
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            float lambda = LAMBDA_MIN + float(i) * lambdaStep;
            float3 cmf = wavelengthToXYZ(lambda);
            
            // Accumulate weighted by the spectrum value and wavelength step
            xyz += spec.samples[i] * cmf * lambdaStep;
        }
        
        // Normalize by CIE Y integral (106.856895)
        return xyz / 106.856895f;
    }

    [mutating]
    void fromXYZ(float3 xyz)
    {
        // Similar to fromRGB but starting from XYZ
        float lambdaStep = (LAMBDA_MAX - LAMBDA_MIN) / float(NUM_SAMPLES - 1);
        
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            float lambda = LAMBDA_MIN + float(i) * lambdaStep;
            
            // Get XYZ color matching functions at this wavelength
            float3 cmf = wavelengthToXYZ(lambda);
            
            // Reconstruct spectral sample
            this.samples[i] = dot(xyz, cmf);
        }
    }

    float3 toFloat3()
    {
        // Convert spectrum to XYZ, then to RGB
        float3 xyz = spectrumToXYZ(this);
        return XYZtoRGB(xyz);
    }
};

// Helper functions for SampledSpectrum operations
SampledSpectrum add(SampledSpectrum a, SampledSpectrum b)
{
    SampledSpectrum result = SampledSpectrum(0.0);
    for (int i = 0; i < 8; ++i) {
        result.samples[i] = a.samples[i] + b.samples[i];
    }
    return result;
}

SampledSpectrum sub(SampledSpectrum a, SampledSpectrum b)
{
    SampledSpectrum result = SampledSpectrum(0.0);
    for (int i = 0; i < 8; ++i) {
        result.samples[i] = a.samples[i] - b.samples[i];
    }
    return result;
}

SampledSpectrum mul(SampledSpectrum a, SampledSpectrum b)
{
    SampledSpectrum result = SampledSpectrum(0.0);
    for (int i = 0; i < 8; ++i) {
        result.samples[i] = a.samples[i] * b.samples[i];
    }
    return result;
}

SampledSpectrum mul(SampledSpectrum a, float scalar)
{
    SampledSpectrum result = SampledSpectrum(0.0);
    for (int i = 0; i < 8; ++i) {
        result.samples[i] = a.samples[i] * scalar;
    }
    return result;
}

SampledSpectrum mul(float scalar, SampledSpectrum a)
{
    return mul(a, scalar);
}

SampledSpectrum div(SampledSpectrum a, float scalar)
{
    SampledSpectrum result = SampledSpectrum(0.0);
    for (int i = 0; i < 8; ++i) {
        result.samples[i] = a.samples[i] / scalar;
    }
    return result;
}

// ============================================================================
// Spectrum Type Selection
// ============================================================================
#ifdef USE_SAMPLED_SPECTRUM
typedef SampledSpectrum Spectrum;
#else
typedef RGBSpectrum Spectrum;
#endif

#endif 