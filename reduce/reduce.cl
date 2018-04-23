__kernel void gpu_reduce_gmem(__global int * g_in, __global int * g_out)
{
   size_t idx        = get_global_id (0);
   size_t thread_idx = get_local_id  (0);
   size_t block_size = get_local_size(0);

   for(size_t s = 1; s < block_size; s *= 2)
   {
      if (thread_idx % (2 * s) == 0)
         g_in[idx] += g_in[idx + s];
      barrier(CLK_GLOBAL_MEM_FENCE);
   }
   if(thread_idx == 0) g_out[get_group_id(0)] = g_in[idx];
}

__kernel void gpu_reduce_lmem(__global int * g_in, __global int * g_out, __local int * sdata)
{
   size_t idx        = get_global_id (0);
   size_t thread_idx = get_local_id  (0);
   size_t block_size = get_local_size(0);

   //Каждый поток загружает один элемент из глобальной памяти в локальную
   sdata[thread_idx] = g_in[idx];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(size_t s = 1; s < block_size; s *= 2)
   {
      if (thread_idx % (2 * s) == 0)
         sdata[thread_idx] += sdata[thread_idx + s];
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   if(thread_idx == 0) g_out[get_group_id(0)] = sdata[0];
}

__kernel void gpu_reduce_lmem_interleaved_addressing(__global int * g_in, __global int * g_out, __local int * sdata)
{
   size_t idx        = get_global_id (0);
   size_t thread_idx = get_local_id  (0);
   size_t block_size = get_local_size(0);

   sdata[thread_idx] = g_in[idx];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(size_t s = 1; s < block_size; s *= 2)
   {
      int index = 2 * s * thread_idx;
      if (index < block_size)
         sdata[index] += sdata[index + s];
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   if(thread_idx == 0) g_out[get_group_id(0)] = sdata[0];
}

__kernel void gpu_reduce_lmem_no_banks_conflicts(__global int * g_in, __global int * g_out, __local int * sdata)
{
   size_t idx        = get_global_id (0);
   size_t thread_idx = get_local_id  (0);
   size_t block_size = get_local_size(0);

   sdata[thread_idx] = g_in[idx];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(size_t s = block_size / 2; s > 0; s /= 2)
   {
      if (thread_idx < s)
         sdata[thread_idx] += sdata[thread_idx + s];
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   if(thread_idx == 0) g_out[get_group_id(0)] = sdata[0];
}

__kernel void gpu_reduce_lmem_pragma_unroll(__global int * g_in, __global int * g_out, __local int * sdata)
{
   size_t idx        = get_global_id (0);
   size_t thread_idx = get_local_id  (0);
   size_t block_size = get_local_size(0);

   //Каждый поток загружает один элемент из глобальной памяти в локальную
   sdata[thread_idx] = g_in[idx];
   barrier(CLK_LOCAL_MEM_FENCE);

   #pragma unroll
   for(size_t s = BLOCK_SIZE / 2; s > 1; s /= 2)
   {
      if (thread_idx < s)
         sdata[thread_idx] += sdata[thread_idx + s];
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   if (thread_idx < 1)
         sdata[thread_idx] += sdata[thread_idx + 1];
   if(thread_idx == 0) g_out[get_group_id(0)] = sdata[0];
}

//for NVidia warp >= 32 only
__kernel void gpu_reduce_lmem_unroll(__global int * g_in, __global int * g_out, __local int * sdata)
{
   size_t idx        = get_global_id (0);
   size_t thread_idx = get_local_id  (0);
   size_t block_size = get_local_size(0);

   //Каждый поток загружает один элемент из глобальной памяти в локальную
   sdata[thread_idx] = g_in[idx];
   barrier(CLK_LOCAL_MEM_FENCE);

   #if BLOCK_SIZE >= 512 //Считаем, что размер блока не более 512
   if (thread_idx < 256) {sdata[thread_idx] += sdata[thread_idx + 256];} barrier(CLK_LOCAL_MEM_FENCE);
   #endif
   #if BLOCK_SIZE >= 256
   if (thread_idx < 128) {sdata[thread_idx] += sdata[thread_idx + 128];} barrier(CLK_LOCAL_MEM_FENCE);
   #endif
   #if BLOCK_SIZE >= 128
   if (thread_idx < 64) {sdata[thread_idx] += sdata[thread_idx + 64];} barrier(CLK_LOCAL_MEM_FENCE);
   #endif

   volatile __local int * v_sdata = sdata; //volatile!!!

   #if BLOCK_SIZE >= 64 
      if (thread_idx < 32) { v_sdata[thread_idx] += v_sdata[thread_idx + 32]; }
   #endif
   #if (BLOCK_SIZE >= 32)
      if (thread_idx < 16) { v_sdata[thread_idx] += v_sdata[thread_idx + 16]; }
   #endif
   #if (BLOCK_SIZE >= 16)
      if (thread_idx < 8) { v_sdata[thread_idx] += v_sdata[thread_idx + 8]; }
   #endif
   #if (BLOCK_SIZE >= 8)
      if (thread_idx < 4) { v_sdata[thread_idx] += v_sdata[thread_idx + 4]; }
   #endif
   #if (BLOCK_SIZE >= 4) 
      if (thread_idx < 2) { v_sdata[thread_idx] += v_sdata[thread_idx + 2]; }
   #endif
   #if (BLOCK_SIZE >= 2)
      if (thread_idx < 1) {v_sdata[thread_idx] += v_sdata[thread_idx + 1];} 
   #endif
   
   if(thread_idx == 0) g_out[get_group_id(0)] = sdata[0];
}