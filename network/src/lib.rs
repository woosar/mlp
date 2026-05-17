mod data_structures;
mod helpers;
pub use data_structures::activations::{Activation, Activations};
pub use data_structures::batch::Batch;
pub use data_structures::neural_net::NeuralNet;

pub trait BatchProvider {
    fn provide_batch(&mut self) -> Option<Batch>;
    fn reset(&mut self);
}

pub trait Evaluable {
    fn create_inference_batch(&self) -> Batch;
    fn set_prediction(&mut self, batch: Batch);
}

/// Enables Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) modes on x86_64 CPUs.
/// This prevents massive slowdowns when neural network weights/gradients approach zero.
#[cfg(target_arch = "x86_64")]
pub fn enable_fast_math() {
    // massive headache to wrap your head around the fact that the use of store and load is inverted
    // compared to today's use. STORE is into a local variable and LOAD is back into memory. weird.
    // TIL: ASM is from the perspective of the CPU registers
    unsafe {
        // MXCSR control/status register that gets activated for denormal or small numbers
        let mut mxcsr: u32 = 0;

        // FTZ  = bit 15
        // DAZ  = bit 6
        const FTZ_DAZ: u32 = 0x8040;

        std::arch::asm!(
        "stmxcsr [{}]", // store register entry in memory
        in(reg) &mut mxcsr,
        options(nostack, preserves_flags),
        );

        mxcsr |= FTZ_DAZ; // masking

        std::arch::asm!(
        "ldmxcsr [{}]", // load into register
        in(reg) &mxcsr,
        options(nostack, preserves_flags),
        );
    }
}
