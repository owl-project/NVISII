// struct Reservoir {
//   float sample;
//   float w_sum = 0;
//   float M = 0; // number of samples
//   // Compute W using (1/p_hat) * (w_sum/M)
  
//   __both__ void update(
//       float x_i, 
//       float w_i, 
//       float rnd,
//       float rnd2) 
//   {
//     w_sum += w_i;
//     M += 1;
//     if (rnd < (w_i / w_sum)) {
//       sample = x_i;
//     }    
//   }

//   __both__ void reset()
//   {
//     M = 0.f;
//     w_sum = 0.f;
//   }

//   __both__ float getWeight(float p_hat) {
//     return (1.f / p_hat) * (w_sum / M);
//   }
// };