#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "windows.h"
#include "lua.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DirectionalBlur.cuh"
#include <iostream>
#include <string>
using namespace std;



int RoundUp(float value, int radix) {
	return (value + radix - 1) / radix;
}

__global__ void Blur_Core(float *r0, float *g0, float *b0, float *a0, int w, int h, int l, int sp, int ow, int mag){
	float ox = (threadIdx.x + blockIdx.x * blockDim.x);
	float oy = (threadIdx.y + blockIdx.y * blockDim.y);
	float rt = 0, gt = 0, bt = 0, at = 0;
//	int ls = floor((float)(l / sp + 0.5));
	float lsx = ((float)l / (float)sp);
	float px = (ox + 0.5f - lsx*sp * mag) / ow, py = (oy + 0.5f) / h;
	float ppx = (float)sp / ow;
	if ((ox < w) && (oy < h)){
		unsigned long off = ox + w * oy;
		float len = 0, alen = 0;
		for (float i = -lsx; i <= lsx; i++){
			float aj = 0;
			aj = tex2D(atex, px, py);
			at += aj;
			alen++;
			if (aj){
				rt += (float)tex2D(rtex, px, py);
				gt += (float)tex2D(gtex, px, py);
				bt += (float)tex2D(btex, px, py);
				len++;
			}
			px += ppx;
		}
		if (len == 0){ len = 1; }
		r0[off] = rt / len;
		g0[off] = gt / len;
		b0[off] = bt / len;
		a0[off] = at / alen; //a0[off] = 255;

	}
}
__global__ void Blur_Gaussian_Core(float *r0, float *g0, float *b0, float *a0, int w, int h, int l, int sp, int ow, int mag){
	float ox = (threadIdx.x + blockIdx.x * blockDim.x);
	float oy = (threadIdx.y + blockIdx.y * blockDim.y);
	float rt = 0, gt = 0, bt = 0, at = 0;
//	int ls = floor((float)(l / sp + 0.5));
	float lsx = ((float)l / (float)sp);
	float px = (ox + 0.5f - lsx*sp * mag) / ow, py = (oy + 0.5f) / h;
	float ppx = (float)sp / (float)ow;
	float sigma = lsx / 2.5f;

	if ((ox < w) && (oy < h)){
		unsigned long off = ox + w * oy;
		float len = 0, alen = 0;
		for (float i = -lsx; i <= lsx; i++){
			float gauss = exp(-(i*i) / (2 * sigma*sigma));
			float aj = 0;
			aj = tex2D(atex, px, py);
			at += aj*gauss;
			alen += gauss;
			if (ceil(aj)!=0.0f){
				rt += (float)tex2D(rtex, px, py)*gauss;
				gt += (float)tex2D(gtex, px, py)*gauss;
				bt += (float)tex2D(btex, px, py)*gauss;
				len += gauss;
			}
			px += ppx;
		}
		if (len == 0){ len = 1; }
		r0[off] = rt/len;
		g0[off] = gt/len;
		b0[off] = bt/len;
		a0[off] = at/alen;

	}
}

__global__ void Separate(unsigned long *data, float *rt, float *gt, float *bt, float *at, int w, int h, int sw, int sh){
	long ox = blockIdx.x;
	long oy = blockIdx.y;
	long ox2 = blockIdx.x+sw;
	long oy2 = blockIdx.y+sh;
	unsigned long datap;
	if (ox<w && oy<h){
		unsigned long i0 = ox + gridDim.x * blockDim.x * oy;
		unsigned long i = ox2 + (w + sw*2) * (oy2);
		datap = data[i0];
		rt[i] = (((datap >> 16) & 0xff));
		gt[i] = (((datap >> 8) & 0xff));
		bt[i] = ((datap & 0xff));
		at[i] = (float)((datap >> 24) & 0xff);
	}
}

__global__ void Rotation(float *r, float *g, float *b, float *a, int w, int h, float sin0, float cos0, int ow, int oh, float div){
	float ox = (threadIdx.x + blockIdx.x * blockDim.x);
	float oy = (threadIdx.y + blockIdx.y * blockDim.y);
	float rx = ox - w / 2.f + 0.5;
	float ry = oy - h / 2.f + 0.5;
	float posx = rx * cos0 - ry * sin0 + ow / (2.f/div) + 0;
	float posy = rx * sin0 + ry * cos0 + oh / (2.f/div) + 0;
	posx /= (float)ow*div;
	posy /= (float)oh*div;

	if (ox < w && oy < h){
		unsigned long off = ox + w * oy;
		r[off] = tex2D(rtex, posx, posy);
		g[off] = tex2D(gtex, posx, posy);
		b[off] = tex2D(btex, posx, posy);
		a[off] = tex2D(atex, posx, posy);

	}

}
__global__ void RotationR(unsigned long *data, int w, int h, float sin0, float cos0, int ow, int oh, float div){
	float ox = (threadIdx.x + blockIdx.x * blockDim.x);
	float oy = (threadIdx.y + blockIdx.y * blockDim.y);
	float rx = ox - w / 2.f + 0.5;
	float ry = oy - h / 2.f + 0.5;
	float posx = rx * cos0 - ry * sin0 + ow / (2.f * div) + 0;
	float posy = rx * sin0 + ry * cos0 + oh / (2.f * div) + 0;
	posx /= (float)ow/div;
	posy /= (float)oh/div;
	unsigned char r = 0, g = 0, b = 0, a = 0;
	if (ox < w && oy < h){
		unsigned long off = ox + w * oy;
		r = tex2D(rtex, posx, posy);
		g = tex2D(gtex, posx, posy);
		b = tex2D(btex, posx, posy);
		a = tex2D(atex, posx, posy);
		data[off] = (r << 16) | (g << 8) | b | (a << 24);
	}

}

/*__global__ void a0(unsigned long *data, float *rt, float *gt, float *bt, float *at, int w, int h){
	float ox = (threadIdx.x + blockIdx.x * blockDim.x);
	float oy = (threadIdx.y + blockIdx.y * blockDim.y);
	unsigned char r, g, b, a;
	if (ox < w && oy < h){
		unsigned long off = ox + w * oy;
		r = floor((rt[off]) + 0.5);
		g = floor((gt[off]) + 0.5);
		b = floor((bt[off]) + 0.5);
		a = floor((at[off]) + 0.5);
		data[off] = (r << 16) | (g << 8) | b | (a << 24);

	}
}
__global__ void b0(unsigned long *data, int w, int h){
	float ox = (threadIdx.x + blockIdx.x * blockDim.x);
	float oy = (threadIdx.y + blockIdx.y * blockDim.y);
	float px = (ox+0.5f) / w, py = (oy+0.5f) / h;
	unsigned char r, g, b, a;
	if (ox < w && oy < h){
		unsigned long off = ox + w * oy;
		r = floor(tex2D(rtex, px, py) + 0.5);
		g = floor(tex2D(gtex, px, py) + 0.5);
		b = floor(tex2D(btex, px, py) + 0.5);
		a = floor(tex2D(atex, px, py) + 0.5);
		data[off] = (r << 16) | (g << 8) | b | (a << 24);

	}
}
*/

int DirectionalBlur_Direct(lua_State *L){
	// 画像データ、幅、高さ等のパラメータを取得
//	unsigned long *data = (unsigned long*)lua_touserdata(L, 1);
//	int w = (int)lua_tonumber(L, 2);
//	int h = (int)lua_tonumber(L, 3);
	int sl = 2;
	int gmd = 0;
	int rm = 0;
	int mm = 0;
	int ds = 1;
	int an = lua_gettop(L);

	int l = (int)lua_tonumber(L, 1);
	float deg = (float)lua_tonumber(L, 2) + 90;
	if (an >= 3) ds = (int)lua_tonumber(L, 3);
	if (an >= 4) sl = (int)lua_tonumber(L, 4);
	if (an >= 5) gmd = (int)lua_tonumber(L, 5);
	if (an >= 6) rm = (int)lua_tonumber(L, 6);
	if (an >= 7) mm = (int)lua_tonumber(L, 7);


	lua_getglobal(L, "obj");
	lua_getfield(L, -1, "getinfo");
	lua_pushstring(L, "image_max");
	lua_call(L, 1, 2);
	int maxh = lua_tointeger(L, -1);
	lua_pop(L, 1);
	int maxw = lua_tointeger(L, -1);
	lua_pop(L, 1);

	lua_getfield(L, -1, "getpixeldata");
	lua_call(L, 0, 3);
	int h = lua_tointeger(L, -1);
	lua_pop(L, 1);
	int w = lua_tointeger(L, -1);
	lua_pop(L, 1);
	unsigned long *data = (unsigned long*)lua_touserdata(L, -1);
	lua_pop(L, 1);

	float rad = deg * 3.141592 / 180.0;
	float cos0 = cos(rad);
	float sin0 = sin(rad);
	float rrad = -deg * 3.141592 / 180.0;
	float rcos0 = cos(rrad);
	float rsin0 = sin(rrad);
	int dw = w;
	int dh = h;
	dw = (int)(floor((w * abs(cos0)) + (h * abs(sin0)) + 0.5));
	dh = (int)(floor((w * abs(sin0)) + (h * abs(cos0)) + 0.5));
	int sw = 0;
	int sh = 0;
	int nw = w;
	int nh = h;
	float div = 1.f;
	if (ds != 0){
		if (l > 50 && l <= 150){
			div = 1.f / (1 + ((l - 50) / 100.f));
		} else if (l > 150){
			div = 1.f / 2.f;
		}
	}
	float tl = l;
	l = floor(l*div+0.5);
	int dbw = dw + l * 2/div;
	if (rm == 0 && mm == 0){
		sw = abs(ceil(tl * 1 * cos0));
		sh = abs(ceil(tl * 1 * sin0));
		nw = w + sw * 2;
		nh = h + sh * 2;
	}
	int dbw0 = dw;
	int mag = 2;
	if (mm != 0){
		dbw0 = dbw;
		mag = 1;
	}
	if (nw > maxw) nw = maxw;
	if (nh > maxh) nh = maxh;
	dbw0 = floor(dbw0*div + 0.5), dbw = floor(dbw*div + 0.5), dh = floor(dh*div + 0.5);
//	nw = dbw0, nh = dh;


	if (sl<=0){ sl = 1; }
	int sp = 1;
	if (l > 50){ sp = sl; }
//	float wa = 1.f / w, ha = 1.f / h;
	if (l != 0){
		int bw = RoundUp(dw, 32);
		int bh = RoundUp(dh, 32);
		dim3 block(bw, bh);
		dim3 block2(RoundUp(dbw, 32), bh);
		dim3 blockn(RoundUp(nw, 32), RoundUp(nh, 32));
		dim3 th(32, 32);
		dim3 tm(w, h);
		unsigned long *datat, *data2;
		float *rt, *gt, *bt, *at;
		float *rt2, *gt2, *bt2, *at2;
		float *rt3, *gt3, *bt3, *at3;
		cudaArray *r_array, *g_array, *b_array, *a_array;
		cudaArray *r2_array, *g2_array, *b2_array, *a2_array;
		cudaArray *r3_array, *g3_array, *b3_array, *a3_array;
		cudaChannelFormatDesc cd = cudaCreateChannelDesc<float>();
		cudaMalloc((void**)&datat, sizeof(unsigned long)*nw*nh);
		cudaMalloc((void**)&data2, sizeof(unsigned long)*w*h);
		cudaMalloc((void**)&rt, sizeof(float)*w*h);
		cudaMalloc((void**)&gt, sizeof(float)*w*h);
		cudaMalloc((void**)&bt, sizeof(float)*w*h);
		cudaMalloc((void**)&at, sizeof(float)*w*h);
		cudaMalloc((void**)&rt2, sizeof(float)*dbw0*dh);
		cudaMalloc((void**)&gt2, sizeof(float)*dbw0*dh);
		cudaMalloc((void**)&bt2, sizeof(float)*dbw0*dh);
		cudaMalloc((void**)&at2, sizeof(float)*dbw0*dh);
		cudaMalloc((void**)&rt3, sizeof(float)*dbw*dh);
		cudaMalloc((void**)&gt3, sizeof(float)*dbw*dh);
		cudaMalloc((void**)&bt3, sizeof(float)*dbw*dh);
		cudaMalloc((void**)&at3, sizeof(float)*dbw*dh);
		cudaMallocArray(&r_array, &cd, w, h);
		cudaMallocArray(&g_array, &cd, w, h);
		cudaMallocArray(&b_array, &cd, w, h);
		cudaMallocArray(&a_array, &cd, w, h);
		cudaMallocArray(&r2_array, &cd, dbw0, dh);
		cudaMallocArray(&g2_array, &cd, dbw0, dh);
		cudaMallocArray(&b2_array, &cd, dbw0, dh);
		cudaMallocArray(&a2_array, &cd, dbw0, dh);
		cudaMallocArray(&r3_array, &cd, dbw, dh);
		cudaMallocArray(&g3_array, &cd, dbw, dh);
		cudaMallocArray(&b3_array, &cd, dbw, dh);
		cudaMallocArray(&a3_array, &cd, dbw, dh);
		cudaTextureAddressMode add = cudaAddressModeMirror;
		cudaTextureAddressMode aadd = cudaAddressModeBorder;
		cudaTextureFilterMode fil = cudaFilterModeLinear;
		if (mm != 0){ aadd = cudaAddressModeMirror; }
		int nor = 1;
		rtex.addressMode[0] = add;
		rtex.addressMode[1] = add;
		rtex.filterMode = fil;
		rtex.normalized = nor;
		gtex.addressMode[0] = add;
		gtex.addressMode[1] = add;
		gtex.filterMode = fil;
		gtex.normalized = nor;
		btex.addressMode[0] = add;
		btex.addressMode[1] = add;
		btex.filterMode = fil;
		btex.normalized = nor;
		atex.addressMode[0] = aadd;
		atex.addressMode[1] = aadd;
		atex.filterMode = fil;
		atex.normalized = nor;
		unsigned long *datan = (unsigned long*)lua_newuserdata(L, sizeof(unsigned long)*nw*nh);

		cudaMemcpy(data2, data, sizeof(unsigned long)*w*h, cudaMemcpyHostToDevice);

		Separate <<< tm, 1 >>>(data2, rt, gt, bt, at, w, h, 0, 0);

		cudaMemcpyToArray(r_array, 0, 0, rt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(g_array, 0, 0, gt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(b_array, 0, 0, bt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(a_array, 0, 0, at, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);

		cudaBindTextureToArray(rtex, r_array, cd);
		cudaBindTextureToArray(gtex, g_array, cd);
		cudaBindTextureToArray(btex, b_array, cd);
		cudaBindTextureToArray(atex, a_array, cd);

		Rotation <<<block2, th >>> (rt2, gt2, bt2, at2, dbw0, dh, sin0, cos0, w, h, div);
		
		cudaMemcpyToArray(r2_array, 0, 0, rt2, sizeof(float)*dbw0*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(g2_array, 0, 0, gt2, sizeof(float)*dbw0*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(b2_array, 0, 0, bt2, sizeof(float)*dbw0*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(a2_array, 0, 0, at2, sizeof(float)*dbw0*dh, cudaMemcpyDeviceToDevice);
		cudaBindTextureToArray(rtex, r2_array, cd);
		cudaBindTextureToArray(gtex, g2_array, cd);
		cudaBindTextureToArray(btex, b2_array, cd);
		cudaBindTextureToArray(atex, a2_array, cd);
		if (gmd != 0){
			Blur_Gaussian_Core <<<block2, th >>>(rt3, gt3, bt3, at3, dbw, dh, l, sp, dbw0, mag);
		} else {
					 Blur_Core <<<block2, th >>>(rt3, gt3, bt3, at3, dbw, dh, l, sp, dbw0, mag);
		}
		cudaMemcpyToArray(r3_array, 0, 0, rt3, sizeof(float)*dbw*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(g3_array, 0, 0, gt3, sizeof(float)*dbw*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(b3_array, 0, 0, bt3, sizeof(float)*dbw*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(a3_array, 0, 0, at3, sizeof(float)*dbw*dh, cudaMemcpyDeviceToDevice);
		cudaBindTextureToArray(rtex, r3_array, cd);
		cudaBindTextureToArray(gtex, g3_array, cd);
		cudaBindTextureToArray(btex, b3_array, cd);
		cudaBindTextureToArray(atex, a3_array, cd);
//		DirectionalBlur_Core << < block, th >> > (datat, dw, dh, l, 90.f, sl, sin0, cos0, 1.f/dw, 1.f/dh, 1.f*cos0, 1.f*sin0);
//		b0 << <block2, th >> >(datat, dbw0, dh);
		RotationR<<<blockn, th>>>(datat, nw, nh, rsin0, rcos0, dbw, dh, div);
		

		cudaMemcpy(datan, datat, sizeof(unsigned long)*nw*nh, cudaMemcpyDeviceToHost);

		if (rm == 0 && mm == 0){
			lua_getglobal(L, "obj");
			lua_getfield(L, -1, "effect");
			lua_pushstring(L, "リサイズ");
			lua_pushstring(L, "X");
			lua_pushinteger(L, nw);
			lua_pushstring(L, "Y");
			lua_pushinteger(L, nh);
			lua_pushstring(L, "ドット数でサイズ指定");
			lua_pushinteger(L, 1);
			lua_call(L, 7, 0);
		}

		lua_getglobal(L, "obj");
		lua_getfield(L, -1, "putpixeldata");
		lua_pushlightuserdata(L, datan);
		lua_call(L, 1, 0);


		cudaUnbindTexture(rtex);
		cudaUnbindTexture(gtex);
		cudaUnbindTexture(btex);
		cudaUnbindTexture(atex);
		cudaFreeArray(r_array);
		cudaFreeArray(g_array);
		cudaFreeArray(b_array);
		cudaFreeArray(a_array);
		cudaFreeArray(r2_array);
		cudaFreeArray(g2_array);
		cudaFreeArray(b2_array);
		cudaFreeArray(a2_array);
		cudaFreeArray(r3_array);
		cudaFreeArray(g3_array);
		cudaFreeArray(b3_array);
		cudaFreeArray(a3_array);
		cudaFree(datat);
		cudaFree(data2);
		cudaFree(rt);
		cudaFree(gt);
		cudaFree(bt);
		cudaFree(at);
		cudaFree(rt2);
		cudaFree(gt2);
		cudaFree(bt2);
		cudaFree(at2);
		cudaFree(rt3);
		cudaFree(gt3);
		cudaFree(bt3);
		cudaFree(at3);
		/*	free(ra);
			free(ga);
			free(ba);
			free(aa);
			free(data);*/
		/*	for (int i = 0; i < w*h; i++)
			{
			int ra0 = (((data[i] >> 16) & 0xff));
			int ga0 = (((data[i] >> 8) & 0xff));
			int ba0 = ((data[i] & 0xff));
			int aa0 = at[i];
			char *str;
			OutputDebugString(itoa(aa0, str, 10));
			}*/
			
	}
//	lua_pushinteger(L, nw);
//	lua_pushinteger(L, nh);
	return 0;
	// Lua 側での戻り値の個数を返す(data だけを返すので 1)
}
int DirectionalBlur(lua_State *L){
	// 画像データ、幅、高さ等のパラメータを取得
	int sl = 2;
	int gmd = 0;
	int rm = 0;
	int mm = 0;
	int ds = 1;
	int an = lua_gettop(L);

	unsigned long *data = (unsigned long*)lua_touserdata(L, 1);
	int w = (int)lua_tonumber(L, 2);
	int h = (int)lua_tonumber(L, 3);
	int l = (int)lua_tonumber(L, 4);
	float deg = (float)lua_tonumber(L, 5) + 90;
	if (an >= 6) ds = (int)lua_tonumber(L, 6);
	if (an >= 7) sl = (int)lua_tonumber(L, 7);
	if (an >= 8) gmd = (int)lua_tonumber(L, 8);
	if (an >= 9) rm = (int)lua_tonumber(L, 9);
	if (an >= 10) mm = (int)lua_tonumber(L, 10);

	OutputDebugString(to_string((int)lua_gettop(L)).c_str());

	lua_getglobal(L, "obj");
	lua_getfield(L, -1, "getinfo");
	lua_pushstring(L, "image_max");
	lua_call(L, 1, 2);
	int maxh = lua_tointeger(L, -1);
	lua_pop(L, 1);
	int maxw = lua_tointeger(L, -1);
	lua_pop(L, 1);


	float rad = deg * 3.141592 / 180.0;
	float cos0 = cos(rad);
	float sin0 = sin(rad);
	float rrad = -deg * 3.141592 / 180.0;
	float rcos0 = cos(rrad);
	float rsin0 = sin(rrad);
	int dw = w;
	int dh = h;
	dw = (int)(floor((w * abs(cos0)) + (h * abs(sin0)) + 0.5));
	dh = (int)(floor((w * abs(sin0)) + (h * abs(cos0)) + 0.5));
	int sw = 0;
	int sh = 0;
	int nw = w;
	int nh = h;
	float div = 1.f;
	if (ds != 0){
		if (l > 50 && l <= 150){
			div = 1.f / (1 + ((l - 50) / 100.f));
		}
		else if (l > 150){
			div = 1.f / 2.f;
		}
	}
	float tl = l;
	l = floor(l*div + 0.5);
	int dbw = dw + l * 2 / div;
	if (rm == 0 && mm == 0){
		sw = abs(ceil(tl * 1 * cos0));
		sh = abs(ceil(tl * 1 * sin0));
		nw = w + sw * 2;
		nh = h + sh * 2;
	}
	int dbw0 = dw;
	int mag = 2;
	if (mm != 0){
		dbw0 = dbw;
		mag = 1;
	}
	if (nw > maxw) nw = maxw;
	if (nh > maxh) nh = maxh;
	dbw0 = floor(dbw0*div + 0.5), dbw = floor(dbw*div + 0.5), dh = floor(dh*div + 0.5);
	//	nw = dbw0, nh = dh;


	if (sl <= 0){ sl = 1; }
	int sp = 1;
	if (l > 50){ sp = sl; }
	//	float wa = 1.f / w, ha = 1.f / h;
	if (l != 0){
		int bw = RoundUp(dw, 32);
		int bh = RoundUp(dh, 32);
		dim3 block(bw, bh);
		dim3 block2(RoundUp(dbw, 32), bh);
		dim3 blockn(RoundUp(nw, 32), RoundUp(nh, 32));
		dim3 th(32, 32);
		dim3 tm(w, h);
		unsigned long *datat, *data2;
		float *rt, *gt, *bt, *at;
		float *rt2, *gt2, *bt2, *at2;
		float *rt3, *gt3, *bt3, *at3;
		cudaArray *r_array, *g_array, *b_array, *a_array;
		cudaArray *r2_array, *g2_array, *b2_array, *a2_array;
		cudaArray *r3_array, *g3_array, *b3_array, *a3_array;
		cudaChannelFormatDesc cd = cudaCreateChannelDesc<float>();
		cudaMalloc((void**)&datat, sizeof(unsigned long)*nw*nh);
		cudaMalloc((void**)&data2, sizeof(unsigned long)*w*h);
		cudaMalloc((void**)&rt, sizeof(float)*w*h);
		cudaMalloc((void**)&gt, sizeof(float)*w*h);
		cudaMalloc((void**)&bt, sizeof(float)*w*h);
		cudaMalloc((void**)&at, sizeof(float)*w*h);
		cudaMalloc((void**)&rt2, sizeof(float)*dbw0*dh);
		cudaMalloc((void**)&gt2, sizeof(float)*dbw0*dh);
		cudaMalloc((void**)&bt2, sizeof(float)*dbw0*dh);
		cudaMalloc((void**)&at2, sizeof(float)*dbw0*dh);
		cudaMalloc((void**)&rt3, sizeof(float)*dbw*dh);
		cudaMalloc((void**)&gt3, sizeof(float)*dbw*dh);
		cudaMalloc((void**)&bt3, sizeof(float)*dbw*dh);
		cudaMalloc((void**)&at3, sizeof(float)*dbw*dh);
		cudaMallocArray(&r_array, &cd, w, h);
		cudaMallocArray(&g_array, &cd, w, h);
		cudaMallocArray(&b_array, &cd, w, h);
		cudaMallocArray(&a_array, &cd, w, h);
		cudaMallocArray(&r2_array, &cd, dbw0, dh);
		cudaMallocArray(&g2_array, &cd, dbw0, dh);
		cudaMallocArray(&b2_array, &cd, dbw0, dh);
		cudaMallocArray(&a2_array, &cd, dbw0, dh);
		cudaMallocArray(&r3_array, &cd, dbw, dh);
		cudaMallocArray(&g3_array, &cd, dbw, dh);
		cudaMallocArray(&b3_array, &cd, dbw, dh);
		cudaMallocArray(&a3_array, &cd, dbw, dh);
		cudaTextureAddressMode add = cudaAddressModeMirror;
		cudaTextureAddressMode aadd = cudaAddressModeBorder;
		cudaTextureFilterMode fil = cudaFilterModeLinear;
		if (mm != 0){ aadd = cudaAddressModeMirror; }
		int nor = 1;
		rtex.addressMode[0] = add;
		rtex.addressMode[1] = add;
		rtex.filterMode = fil;
		rtex.normalized = nor;
		gtex.addressMode[0] = add;
		gtex.addressMode[1] = add;
		gtex.filterMode = fil;
		gtex.normalized = nor;
		btex.addressMode[0] = add;
		btex.addressMode[1] = add;
		btex.filterMode = fil;
		btex.normalized = nor;
		atex.addressMode[0] = aadd;
		atex.addressMode[1] = aadd;
		atex.filterMode = fil;
		atex.normalized = nor;
		unsigned long *datan = (unsigned long*)lua_newuserdata(L, sizeof(unsigned long)*nw*nh);

		cudaMemcpy(data2, data, sizeof(unsigned long)*w*h, cudaMemcpyHostToDevice);

		Separate << < tm, 1 >> >(data2, rt, gt, bt, at, w, h, 0, 0);

		cudaMemcpyToArray(r_array, 0, 0, rt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(g_array, 0, 0, gt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(b_array, 0, 0, bt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(a_array, 0, 0, at, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);

		cudaBindTextureToArray(rtex, r_array, cd);
		cudaBindTextureToArray(gtex, g_array, cd);
		cudaBindTextureToArray(btex, b_array, cd);
		cudaBindTextureToArray(atex, a_array, cd);

		Rotation << <block2, th >> > (rt2, gt2, bt2, at2, dbw0, dh, sin0, cos0, w, h, div);

		cudaMemcpyToArray(r2_array, 0, 0, rt2, sizeof(float)*dbw0*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(g2_array, 0, 0, gt2, sizeof(float)*dbw0*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(b2_array, 0, 0, bt2, sizeof(float)*dbw0*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(a2_array, 0, 0, at2, sizeof(float)*dbw0*dh, cudaMemcpyDeviceToDevice);
		cudaBindTextureToArray(rtex, r2_array, cd);
		cudaBindTextureToArray(gtex, g2_array, cd);
		cudaBindTextureToArray(btex, b2_array, cd);
		cudaBindTextureToArray(atex, a2_array, cd);
		if (gmd != 0){
			Blur_Gaussian_Core << <block2, th >> >(rt3, gt3, bt3, at3, dbw, dh, l, sp, dbw0, mag);
		}
		else {
			Blur_Core << <block2, th >> >(rt3, gt3, bt3, at3, dbw, dh, l, sp, dbw0, mag);
		}
		cudaMemcpyToArray(r3_array, 0, 0, rt3, sizeof(float)*dbw*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(g3_array, 0, 0, gt3, sizeof(float)*dbw*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(b3_array, 0, 0, bt3, sizeof(float)*dbw*dh, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(a3_array, 0, 0, at3, sizeof(float)*dbw*dh, cudaMemcpyDeviceToDevice);
		cudaBindTextureToArray(rtex, r3_array, cd);
		cudaBindTextureToArray(gtex, g3_array, cd);
		cudaBindTextureToArray(btex, b3_array, cd);
		cudaBindTextureToArray(atex, a3_array, cd);
		//		DirectionalBlur_Core << < block, th >> > (datat, dw, dh, l, 90.f, sl, sin0, cos0, 1.f/dw, 1.f/dh, 1.f*cos0, 1.f*sin0);
		//		b0 << <block2, th >> >(datat, dbw0, dh);
		RotationR << <blockn, th >> >(datat, nw, nh, rsin0, rcos0, dbw, dh, div);


		cudaMemcpy(datan, datat, sizeof(unsigned long)*nw*nh, cudaMemcpyDeviceToHost);


		cudaUnbindTexture(rtex);
		cudaUnbindTexture(gtex);
		cudaUnbindTexture(btex);
		cudaUnbindTexture(atex);
		cudaFreeArray(r_array);
		cudaFreeArray(g_array);
		cudaFreeArray(b_array);
		cudaFreeArray(a_array);
		cudaFreeArray(r2_array);
		cudaFreeArray(g2_array);
		cudaFreeArray(b2_array);
		cudaFreeArray(a2_array);
		cudaFreeArray(r3_array);
		cudaFreeArray(g3_array);
		cudaFreeArray(b3_array);
		cudaFreeArray(a3_array);
		cudaFree(datat);
		cudaFree(data2);
		cudaFree(rt);
		cudaFree(gt);
		cudaFree(bt);
		cudaFree(at);
		cudaFree(rt2);
		cudaFree(gt2);
		cudaFree(bt2);
		cudaFree(at2);
		cudaFree(rt3);
		cudaFree(gt3);
		cudaFree(bt3);
		cudaFree(at3);
		/*	free(ra);
		free(ga);
		free(ba);
		free(aa);
		free(data);*/
		/*	for (int i = 0; i < w*h; i++)
		{
		int ra0 = (((data[i] >> 16) & 0xff));
		int ga0 = (((data[i] >> 8) & 0xff));
		int ba0 = ((data[i] & 0xff));
		int aa0 = at[i];
		char *str;
		OutputDebugString(itoa(aa0, str, 10));
		}*/

	}
	lua_pushinteger(L, nw);
	lua_pushinteger(L, nh);
	return 3;
	// Lua 側での戻り値の個数を返す(data だけを返すので 1)
}

static luaL_Reg DirectionalBlur_s[] = {
	{ "DirectionalBlur", DirectionalBlur },
	{ "DirectionalBlur_Direct", DirectionalBlur_Direct },
	{ NULL, NULL }
};
/*
ここでdllを定義します
別のものを作る場合は
Reverse_s
の部分を新しい名前に変えてください
*/
extern "C"{
	__declspec(dllexport) int luaopen_DirectionalBlur_s(lua_State *L) {
		luaL_register(L, "DirectionalBlur_s", DirectionalBlur_s);
	return 1;
}
}
