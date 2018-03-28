#include "stdafx.h"

using namespace std;

// Structure for the cbuffer data
struct CSConstantBuffer
{
	float PosX;
	float PosY;
	float Zoom;
	uint32_t Iterations;
};

struct CSConstantBuffer_Double
{
	double PosX;
	double PosY;

	double Zoom;
	uint32_t Iterations;
	uint32_t padding;
};


double CurrentPosX = 0;
double CurrentPosY = 0;
double CurrentZoom = 1.0f;
uint32_t CurrentInterations = 100;
bool UseDouble = false;

// Using this instead of the KeyboardCallback because I want it to handle continuous presses
void _bind_key(function<void(bool)> f, int key, int mod_key = -1)
{
	if (GetAsyncKeyState(key) & 0x8000)
	{
		f(mod_key != -1 && (GetAsyncKeyState(mod_key) & 0x8000));
	}
}

#define bind_key(exp, ... ) _bind_key([&](bool bmod){ exp; },__VA_ARGS__)

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
			wcout << L"	-,+ = Zoom" << endl;
			wcout << L"	Arrows for movement" << endl;
			wcout << L"	F1,F2 = Less/More iterations" << endl;
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
			bind_key(CurrentZoom -= 0.05 * CurrentZoom, VK_SUBTRACT);
			bind_key(CurrentZoom += 0.05 * CurrentZoom, VK_ADD);
			bind_key(CurrentPosY -= (bmod) ? 0.1 * CurrentZoom : 0.01 * CurrentZoom, VK_UP, VK_SHIFT);
			bind_key(CurrentPosY += (bmod) ? 0.1 * CurrentZoom : 0.01 * CurrentZoom, VK_DOWN, VK_SHIFT);
			bind_key(CurrentPosX += (bmod) ? 0.1 * CurrentZoom : 0.01 * CurrentZoom, VK_RIGHT, VK_SHIFT);
			bind_key(CurrentPosX -= (bmod) ? 0.1 * CurrentZoom : 0.01 * CurrentZoom, VK_LEFT, VK_SHIFT);
			bind_key(CurrentInterations += (bmod) ? 10 : 1, VK_F1, VK_SHIFT);
			bind_key(CurrentInterations -= (bmod) ? (CurrentInterations >= 10 ? 10 : 0) : (CurrentInterations >= 1 ? 1 : 0), VK_F2, VK_SHIFT);

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
	desc.WindowDescription.SizeX = 1024;
	desc.WindowDescription.SizeY = 1024;
	desc.SwapChainDescription.BackbufferAccessFlags |= DXGI_USAGE_UNORDERED_ACCESS;
	desc.SwapChainDescription.BackbufferAccessFlags |= DXGI_USAGE_SHADER_INPUT;
	LogCheck(dev.Start(desc), FrameDX::LogCategory::CriticalError);

	// Create texture for the cpu side
	FrameDX::Texture2D cpu_texture;
	auto tex_desc = FrameDX::Texture2D::Description();
	tex_desc.SizeX = dev.GetBackbuffer()->Desc.SizeX;
	tex_desc.SizeY = dev.GetBackbuffer()->Desc.SizeY;
	tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

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
	mandelbrot_cs_double.CreateFromFile(&dev, L"Mandelbrot.hlsl", "main", false, { {"USE_DOUBLES","1"} });

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
		if(UseDouble)
		{
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			ZeroMemory(&mappedResource, sizeof(D3D11_MAPPED_SUBRESOURCE));
			dev.GetImmediateContext()->Map(cb_buffer_double, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);

			CSConstantBuffer_Double new_data;
			new_data.PosX = CurrentPosX;
			new_data.PosY = CurrentPosY;
			new_data.Zoom = CurrentZoom;
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
			new_data.PosX = CurrentPosX;
			new_data.PosY = CurrentPosY;
			new_data.Zoom = CurrentZoom;
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
		dev.GetImmediateContext()->Dispatch(ceilf(2*dev.GetBackbuffer()->Desc.SizeX / mandelbrot_cs.GroupSizeX), ceilf(2*dev.GetBackbuffer()->Desc.SizeY / mandelbrot_cs.GroupSizeY), 1);

		dev.GetSwapChain()->Present(0, 0);
	});

	// Not releasing any resources here... Should implement that eventually

	return 0;
}