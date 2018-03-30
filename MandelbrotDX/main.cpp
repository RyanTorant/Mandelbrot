#include "stdafx.h"
#include "WorkerPool.h"
using namespace std;

// Structure for the cbuffer data
struct CSConstantBuffer
{
	float CoeffA_X;
	float CoeffA_Y;
	float CoeffB_X;
	float CoeffB_Y;

	uint32_t Iterations;
	uint32_t padding0;
	uint32_t padding1;
	uint32_t padding2;
};

struct CSConstantBuffer_Double
{
	double CoeffA_X;
	double CoeffA_Y;
	double CoeffB_X;
	double CoeffB_Y;

	uint32_t Iterations;
	uint32_t padding0;
	uint32_t padding1;
	uint32_t padding2;
};

// Using this instead of the KeyboardCallback because I want it to handle continuous presses
void _bind_key(function<void(bool)> f, int key, int mod_key = -1)
{
	if (GetAsyncKeyState(key) & 0x8000)
	{
		f(mod_key != -1 && (GetAsyncKeyState(mod_key) & 0x8000));
	}
}

#define bind_key(exp, ... ) _bind_key([&](bool bmod){ exp; },__VA_ARGS__)

double CurrentPosX = 0;
double CurrentPosY = 0;
double CurrentZoom = 1.0f;
uint32_t CurrentInterations = 100;
bool UseDouble = false;
bool UseCPU = false;

const uint32_t ScreenResX = 1024;
const uint32_t ScreenResY = 1024;
const uint32_t CPUBlockSizeX = 64;
const uint32_t CPUBlockSizeY = 64;
const uint32_t BlocksCountX = FrameDX::ceil(ScreenResX, CPUBlockSizeX);
const uint32_t BlocksCountY = FrameDX::ceil(ScreenResY, CPUBlockSizeY);
const uint32_t TotalBlocksCount = BlocksCountX*BlocksCountY;
const double InvResX = 1.0 / double(ScreenResX);
const double InvResY = 1.0 / double(ScreenResY);

// CPU pooled workers
WorkerPool * Workers = nullptr;

// Computes mandelbrot for the current parameters using multithreading and AVX for x2 SSAA
void CPUMandelbrot(uint8_t * Buffer)
{
	alignas(32) const double mask_x_array[4] = { 0.0, 0.5, 0.0, 0.5 };
	alignas(32) const double mask_y_array[4] = { 0.0, 0.0, 0.5, 0.5 };
	const double coeff_a_x = (4.0*CurrentZoom) * InvResX;
	const double coeff_a_y = (4.0*CurrentZoom) * InvResY;
	const double coeff_b_x = -2.0*CurrentZoom + CurrentPosX;
	const double coeff_b_y = -2.0*CurrentZoom + CurrentPosY;
	const uint64_t bit_mask = 0x3FF0000000000000; // Used to convert the cmp value to 1.0
	const double two = 2.0;
	const double four = 4.0;

	const float zero_f = 0.0f;
	const float one_f = 1.0f;
	const float two_f = 2.0f;
	const float four_f = 4.0f;
	const float six = 6.0f;
	const float three = 3.0f;
	const uint32_t hue_div = 31; // 32 colors, can use & 31 to take mod
	const float hue_mul = 1.0f / 32.0f;
	const uint32_t abs_bit_mask = 0x7FFFFFFF; // Used to mask out the sign bit
	const float final_mul_f = 255.0f * 0.25f; // 255 max per pixel, averaging 4 pixels

	__m256d mask_x_vec = _mm256_load_pd(mask_x_array);
	__m256d mask_y_vec = _mm256_load_pd(mask_y_array);
	__m256d inv_res_x_vec = _mm256_broadcast_sd(&InvResX);
	__m256d inv_res_y_vec = _mm256_broadcast_sd(&InvResY);
	__m256d coeff_a_x_vec = _mm256_broadcast_sd(&coeff_a_x);
	__m256d coeff_a_y_vec = _mm256_broadcast_sd(&coeff_a_y);
	__m256d coeff_b_x_vec = _mm256_broadcast_sd(&coeff_b_x);
	__m256d coeff_b_y_vec = _mm256_broadcast_sd(&coeff_b_y);
	__m256d bit_mask_vec = _mm256_broadcast_sd((double*)&bit_mask);
	__m256d two_vec = _mm256_broadcast_sd(&two);	
	__m256d four_vec = _mm256_broadcast_sd(&four);

	__m128 zero_vec_f = _mm_broadcast_ss(&zero_f);
	__m128 one_vec_f = _mm_broadcast_ss(&one_f);
	__m128 two_vec_f = _mm_broadcast_ss(&two_f);
	__m128 three_vec_f = _mm_broadcast_ss(&three);
	__m128 four_vec_f = _mm_broadcast_ss(&four_f);
	__m128 six_vec_f = _mm_broadcast_ss(&six);
	__m128i hue_div_vec = _mm_set1_epi32(hue_div); // sadly have to use this, no nice broadcast function for ints
	__m128 hue_mul_vec = _mm_broadcast_ss(&hue_mul);
	__m128 abs_bit_mask_vec = _mm_broadcast_ss((float*)&abs_bit_mask);
	__m128i num_iters_vec = _mm_set1_epi32(CurrentInterations);
	__m128 final_mul_vec_f = _mm_broadcast_ss(&final_mul_f);

	Workers->JobSize = TotalBlocksCount;
	Workers->WorkerFunction = [&](int JobIDX, int WorkerIDX)
	{
		int x = uint32_t(JobIDX) % BlocksCountX;
		int y = uint32_t(JobIDX) / BlocksCountX;

		x *= CPUBlockSizeX;
		y *= CPUBlockSizeY;

		for (int s = 0; s < CPUBlockSizeY; s++)
		{
			for (int r = 0; r < CPUBlockSizeX; r++)
			{
				int pX = x + r;
				int pY = y + s;

				if (pX < ScreenResX && pY < ScreenResY)
				{
					double px_d = pX;
					double py_d = pY;

					// Load the pixel positions
					__m256d c_x = _mm256_broadcast_sd(&px_d);
					__m256d c_y = _mm256_broadcast_sd(&py_d);

					// Add the subpixel offsets for SSAA
					c_x = _mm256_add_pd(c_x, mask_x_vec);
					c_y = _mm256_add_pd(c_y, mask_y_vec);

					// Normalize pixel pos
					c_x = _mm256_mul_pd(c_x, coeff_a_x_vec);
					c_y = _mm256_mul_pd(c_y, coeff_a_y_vec);
					c_x = _mm256_add_pd(c_x, coeff_b_x_vec);
					c_y = _mm256_add_pd(c_y, coeff_b_y_vec);

					// Do the iteration
					__m256d z_x = _mm256_setzero_pd();
					__m256d z_y = _mm256_setzero_pd();
					__m256d iters = _mm256_setzero_pd();

					for (float i = 0; i < CurrentInterations; i++)
					{
						// Square
						__m256d x2 = _mm256_mul_pd(z_x, z_x);
						__m256d y2 = _mm256_mul_pd(z_y, z_y);
						
						// Compute y coordinate first
						z_y = _mm256_mul_pd(z_x, z_y);
						z_y = _mm256_mul_pd(z_y, two_vec);

						// Then compute z coordinate
						z_x = _mm256_sub_pd(x2, y2);

						// Add c
						z_x = _mm256_add_pd(z_x,c_x);
						z_y = _mm256_add_pd(z_y,c_y);

						// Compute length
						__m256d x2_r = _mm256_mul_pd(z_x, z_x);
						__m256d y2_r = _mm256_mul_pd(z_y, z_y);
						__m256d r2 = _mm256_add_pd(x2_r, y2_r);

						// Check if length <= 4
						r2 = _mm256_cmp_pd(r2, four_vec, _CMP_LE_OQ);

						// If all are 0, that means they are all r^2 > 4, so end
						if (_mm256_testz_pd(r2,r2))
							break;

						// Now convert the cmp mask to a 1.0 or 0.0
						r2 = _mm256_and_pd(r2, bit_mask_vec);

						// Finally, increase iterations count
						// By using the mask and not just + 1, you only increase the values that are still bounded
						iters = _mm256_add_pd(iters, r2);
					}

					// Now compute rgb color from that
					// Using 32 hues

					// Convert to integer for the mod
					__m128i iters_i = _mm256_cvtpd_epi32(iters);

					// Pixels that are on the set go black
					// No compare != , but have andnot so no problem here
					__m128i in_set_mask = _mm_cmpeq_epi32(iters_i, num_iters_vec);

					// Mod by 32 by doing an & 31
					iters_i = _mm_and_si128(iters_i, hue_div_vec);

					// Convert back to float
					__m128 iters_f = _mm_cvtepi32_ps(iters_i);

					// Normalize and multiply by 6
					iters_f = _mm_mul_ps(iters_f, hue_mul_vec);
					iters_f = _mm_mul_ps(iters_f, six_vec_f);

					// Now start computing each channel
					__m128 r = _mm_sub_ps(iters_f, three_vec_f);
					__m128 g = _mm_sub_ps(iters_f, two_vec_f);
					__m128 b = _mm_sub_ps(iters_f, four_vec_f);

					// mask sign bit because there is no abs inst
					r = _mm_and_ps(r, abs_bit_mask_vec);
					g = _mm_and_ps(g, abs_bit_mask_vec);
					b = _mm_and_ps(b, abs_bit_mask_vec);

					r = _mm_sub_ps(r, one_vec_f);
					g = _mm_sub_ps(two_vec_f, g);
					b = _mm_sub_ps(two_vec_f, b);

					// Clamp each value to [0,1]
					r = _mm_min_ps(_mm_max_ps(r, zero_vec_f), one_vec_f);
					g = _mm_min_ps(_mm_max_ps(g, zero_vec_f), one_vec_f);
					b = _mm_min_ps(_mm_max_ps(b, zero_vec_f), one_vec_f);

					// Mask out the elements that are on the set
					// Using andnot because the comparison is == (don't have !=)
					// The cast is only for the compiler, it doesn't emit any instruction
					r = _mm_castsi128_ps(_mm_andnot_si128(in_set_mask, _mm_castps_si128(r)));
					g = _mm_castsi128_ps(_mm_andnot_si128(in_set_mask, _mm_castps_si128(g)));
					b = _mm_castsi128_ps(_mm_andnot_si128(in_set_mask, _mm_castps_si128(b)));

					// Now average across sub pixels
					//		First multiply by 255.0f * 0.25f = 64.0f
					r = _mm_mul_ps(r, final_mul_vec_f);
					g = _mm_mul_ps(g, final_mul_vec_f);
					b = _mm_mul_ps(b, final_mul_vec_f);

					//		Then reduce
					r = _mm_hadd_ps(r, r);
					g = _mm_hadd_ps(g, g);
					b = _mm_hadd_ps(b, b);

					r = _mm_hadd_ps(r, r);
					g = _mm_hadd_ps(g, g);
					b = _mm_hadd_ps(b, b);

					// Convert to int
					__m128i r_i = _mm_cvtps_epi32(r);
					__m128i g_i = _mm_cvtps_epi32(g);
					__m128i b_i = _mm_cvtps_epi32(b);

					// Finally extract values and store
					// Values are [0,255] anyway, so 16 bits is more than enough
					Buffer[4 * (pX + pY*ScreenResX)    ] = _mm_extract_epi16(r_i, 0);
					Buffer[4 * (pX + pY*ScreenResX) + 1] = _mm_extract_epi16(g_i, 0);
					Buffer[4 * (pX + pY*ScreenResX) + 2] = _mm_extract_epi16(b_i, 0);
					Buffer[4 * (pX + pY*ScreenResX) + 3] = 255;
				}
			}
		}
	};
	
	Workers->Dispatch();
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR, int)
{
	AllocConsole();
	freopen("CONIN$", "r", stdin);
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);

	// Create thread to print to the console screen
	thread log_printer([]()
	{
		FrameDX::TimedLoop([&]()
		{
			system("cls");
			wcout << L"Instructions : " << endl;
			wcout << L"	Q,E = Zoom" << endl;
			wcout << L"	Arrows for movement" << endl;
			wcout << L"	A,D = Less/More iterations" << endl;
			wcout << L"	Shift + any of the above makes them faster" << endl;
			wcout << L"	Press T to toggle double precision (AVX always runs at double precision)" << endl;
			wcout << L"-------------------------------" << endl;
			wcout << L"Zoom : " << CurrentZoom << endl;
			wcout << L"X : " << CurrentPosX << endl;
			wcout << L"Y : " << CurrentPosY << endl;
			wcout << L"n : " << CurrentInterations << endl;
			wcout << ((UseDouble) ? L"Using Double Precision" : L"Using Single Precision") << endl;

			FrameDX::Log.PrintAll(wcout);
		}, 150ms);
	});
	log_printer.detach();

	// Define keyboard callback
	// See remarks on the definition of bind_key
	thread keyboard_handler([&]()
	{
		FrameDX::TimedLoop([&]()
		{
			bind_key(CurrentZoom += ((bmod) ? 0.07 : 0.025) * CurrentZoom, 'Q', VK_SHIFT);
			bind_key(CurrentZoom -= ((bmod) ? 0.07 : 0.025) * CurrentZoom, 'E', VK_SHIFT);
			bind_key(CurrentPosY -= (bmod) ? 0.1 * CurrentZoom : 0.01 * CurrentZoom, VK_UP, VK_SHIFT);
			bind_key(CurrentPosY += (bmod) ? 0.1 * CurrentZoom : 0.01 * CurrentZoom, VK_DOWN, VK_SHIFT);
			bind_key(CurrentPosX += (bmod) ? 0.1 * CurrentZoom : 0.01 * CurrentZoom, VK_RIGHT, VK_SHIFT);
			bind_key(CurrentPosX -= (bmod) ? 0.1 * CurrentZoom : 0.01 * CurrentZoom, VK_LEFT, VK_SHIFT);
			bind_key(CurrentInterations += (bmod) ? 10 : 1, 'D', VK_SHIFT);
			bind_key(CurrentInterations -= (bmod) ? (CurrentInterations >= 10 ? 10 : 0) : (CurrentInterations >= 1 ? 1 : 0), 'A' , VK_SHIFT);

		}, 25ms);
	});

	// Using the callback for the double toggle, as a toggle is not continuous 
	FrameDX::Device::KeyboardCallback = [](WPARAM key, FrameDX::KeyAction action)
	{
		if (key == 'T' && action == FrameDX::KeyAction::Up)
			UseDouble = !UseDouble;
	};

	// Create device
	FrameDX::Device dev;
	auto desc = FrameDX::Device::Description();
	desc.WindowDescription.SizeX = ScreenResX;
	desc.WindowDescription.SizeY = ScreenResY;
	desc.SwapChainDescription.BackbufferAccessFlags |= DXGI_USAGE_UNORDERED_ACCESS;
	desc.SwapChainDescription.BackbufferAccessFlags |= DXGI_USAGE_SHADER_INPUT;
	LogCheck(dev.Start(desc), FrameDX::LogCategory::CriticalError);

	// Create workers
	Workers = new WorkerPool(thread::hardware_concurrency());

	// Create texture for the cpu side
	FrameDX::Texture2D cpu_texture;
	auto tex_desc = FrameDX::Texture2D::Description();
	tex_desc.SizeX = dev.GetBackbuffer()->Desc.SizeX;
	tex_desc.SizeY = dev.GetBackbuffer()->Desc.SizeY;
	tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	tex_desc.Usage = D3D11_USAGE_DYNAMIC;
	tex_desc.AccessFlags = D3D11_CPU_ACCESS_WRITE;

	cpu_texture.CreateFromDescription(&dev, tex_desc);
	
	// Create color table
	// 32 colors
	FrameDX::Texture2D color_table;
	tex_desc = FrameDX::Texture2D::Description();
	tex_desc.SizeX = 32;
	tex_desc.SizeY = 1;
	tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	tex_desc.Usage = D3D11_USAGE_IMMUTABLE;

	vector<uint8_t> img_data(tex_desc.SizeX * 4);
	for (int i = 0; i < tex_desc.SizeX; i++)
	{
		float fi = float(i) / tex_desc.SizeX;

		img_data[4*i] = FrameDX::saturate(abs(fi * 6 - 3) - 1)*255;
		img_data[4*i + 1] = FrameDX::saturate(2 - abs(fi * 6 - 2))*255;
		img_data[4*i + 2] = FrameDX::saturate(2 - abs(fi * 6 - 4))*255;
		img_data[4*i + 3] = 255;
	}
	color_table.CreateFromDescription(&dev, tex_desc, img_data);

	// Load compute shaders
	FrameDX::ComputeShader mandelbrot_cs;
	mandelbrot_cs.CreateFromFile(&dev, L"Mandelbrot.hlsl", "main");
	FrameDX::ComputeShader mandelbrot_cs_double;
	mandelbrot_cs_double.CreateFromFile(&dev, L"Mandelbrot.hlsl", "main", false, { {"USE_DOUBLES","1"}, {"SCREEN_RES_X", to_string(ScreenResX).c_str() }, { "SCREEN_RES_Y", to_string(ScreenResY).c_str() } });

	// Create constant buffer for the cs
	ID3D11Buffer* cb_buffer;
	D3D11_BUFFER_DESC cb_desc;
	cb_desc.ByteWidth = sizeof(CSConstantBuffer);
	cb_desc.Usage = D3D11_USAGE_DYNAMIC;
	cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cb_desc.MiscFlags = 0;
	cb_desc.StructureByteStride = 0;

	LogCheck(dev.GetDevice()->CreateBuffer(&cb_desc, nullptr, &cb_buffer), FrameDX::LogCategory::CriticalError);

	ID3D11Buffer* cb_buffer_double;
	cb_desc.ByteWidth = sizeof(CSConstantBuffer_Double);
	cb_desc.Usage = D3D11_USAGE_DYNAMIC;
	cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cb_desc.MiscFlags = 0;
	cb_desc.StructureByteStride = 0;

	LogCheck(dev.GetDevice()->CreateBuffer(&cb_desc, nullptr, &cb_buffer_double), FrameDX::LogCategory::CriticalError);

	// Create point sampler
	ID3D11SamplerState * point_sampler;
	D3D11_SAMPLER_DESC sampler_desc{};
	sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
	
	LogCheck(dev.GetDevice()->CreateSamplerState(&sampler_desc,&point_sampler), FrameDX::LogCategory::CriticalError);

	// Create pipeline states
	FrameDX::PipelineState cs_state;
	cs_state.Shaders[(size_t)FrameDX::ShaderStage::Compute].ShaderPtr = &mandelbrot_cs;
	cs_state.Shaders[(size_t)FrameDX::ShaderStage::Compute].ConstantBuffersTable = { cb_buffer };
	cs_state.Shaders[(size_t)FrameDX::ShaderStage::Compute].ResourcesTable = { color_table.SRV };
	cs_state.Shaders[(size_t)FrameDX::ShaderStage::Compute].SamplersTable = { point_sampler };
	cs_state.Output.ComputeShaderUAVs = { dev.GetBackbuffer()->UAV };

	FrameDX::PipelineState cs_state_double;
	cs_state_double.Shaders[(size_t)FrameDX::ShaderStage::Compute].ShaderPtr = &mandelbrot_cs_double;
	cs_state_double.Shaders[(size_t)FrameDX::ShaderStage::Compute].ConstantBuffersTable = { cb_buffer_double };
	cs_state_double.Shaders[(size_t)FrameDX::ShaderStage::Compute].ResourcesTable = { color_table.SRV };
	cs_state_double.Shaders[(size_t)FrameDX::ShaderStage::Compute].SamplersTable = { point_sampler };
	cs_state_double.Output.ComputeShaderUAVs = { dev.GetBackbuffer()->UAV };

	dev.EnterMainLoop([&](double GlobalTimeNanoseconds)
	{
		// Update cbuffer
		// TODO : move this to a wrapper function to make it cleaner

		double CoeffA_X = (4.0*CurrentZoom) * InvResX;
		double CoeffA_Y = (4.0*CurrentZoom) * InvResY;
		double CoeffB_X = -2.0*CurrentZoom + CurrentPosX;
		double CoeffB_Y = -2.0*CurrentZoom + CurrentPosY;

		if(UseDouble)
		{
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			ZeroMemory(&mappedResource, sizeof(D3D11_MAPPED_SUBRESOURCE));
			dev.GetImmediateContext()->Map(cb_buffer_double, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);

			CSConstantBuffer_Double new_data;
			new_data.CoeffA_X = CoeffA_X;
			new_data.CoeffA_Y = CoeffA_Y;
			new_data.CoeffB_X = CoeffB_X;
			new_data.CoeffB_Y = CoeffB_Y;
			new_data.Iterations = CurrentInterations;

			memcpy(mappedResource.pData, &new_data, sizeof(CSConstantBuffer_Double));
			dev.GetImmediateContext()->Unmap(cb_buffer_double, 0);
		}
		else
		{
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			ZeroMemory(&mappedResource, sizeof(D3D11_MAPPED_SUBRESOURCE));
			dev.GetImmediateContext()->Map(cb_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);

			CSConstantBuffer new_data;
			new_data.CoeffA_X = CoeffA_X;
			new_data.CoeffA_Y = CoeffA_Y;
			new_data.CoeffB_X = CoeffB_X;
			new_data.CoeffB_Y = CoeffB_Y;
			new_data.Iterations = CurrentInterations;

			memcpy(mappedResource.pData, &new_data, sizeof(CSConstantBuffer));
			dev.GetImmediateContext()->Unmap(cb_buffer, 0);
		}

		// Run compute shader
		// No need to clean as this writes to the whole screen
		if(UseDouble)
			dev.BindPipelineState(cs_state_double);
		else
			dev.BindPipelineState(cs_state);
		dev.GetImmediateContext()->Dispatch(FrameDX::ceil(2*dev.GetBackbuffer()->Desc.SizeX,mandelbrot_cs.GroupSizeX), FrameDX::ceil(2*dev.GetBackbuffer()->Desc.SizeY, mandelbrot_cs.GroupSizeY), 1);




		// DEBUG
		{
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			ZeroMemory(&mappedResource, sizeof(D3D11_MAPPED_SUBRESOURCE));
			dev.GetImmediateContext()->Map(cpu_texture.GetResource(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);

			CPUMandelbrot((uint8_t*)mappedResource.pData);

			dev.GetImmediateContext()->Unmap(cpu_texture.GetResource(), 0);
		}
		dev.GetBackbuffer()->CopyFrom(&cpu_texture);





		dev.GetSwapChain()->Present(0, 0);
	});

	// Not releasing any resources here... Should implement that eventually
	delete Workers;

	return 0;
}