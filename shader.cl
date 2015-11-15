__kernel void adder(__global const int* a, __global const int* b, __global int* result)
{
	int idx = get_global_id(0);
	result[idx] = a[idx] + b[idx];
}