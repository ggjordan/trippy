//! Turning a raw `CubeTensor` into a user-facing `burn::Tensor<D>`.
//!
//! Module: `brush_pyramid::gpu::burn_bridge`
//! Purpose: [`crate::gpu::render_pyramid`] composites into flat, device-
//!     resident `CubeTensor<WgpuRuntime>` buffers, which is what a viewer
//!     wants to bind — but what the U-Net (`brush-unet`) needs is a
//!     `burn::Tensor<4>` it can feed to `Conv2d`. This module is the only
//!     bridge between the two, and the reason `docs/LIMITATIONS.md`'s
//!     "no `Tensor<4>` yet" entry can be closed.
//!
//! # Why it takes an `Operation` registration
//!
//! In the Burn revision this workspace pins (`b6e27bdc`), `Tensor<const D>`
//! is backend-erased over the **`Dispatch`** backend, whose wgpu variant is
//! the **fusion** backend `Fusion<CubeBackend<WgpuRuntime>>`. A fusion
//! tensor is not a buffer: it is a *handle plus a position in a lazily
//! recorded operation stream*. There is no `from_primitive(CubeTensor)`,
//! because a bare buffer has no place in that stream.
//!
//! The supported way in is to register a custom operation whose *inputs are
//! empty* and whose single output the operation binds to an already-computed
//! concrete tensor ([`burn_ir::HandleContainer::register_float_tensor`]).
//! When the stream drains, the bind runs, the handle resolves to our buffer,
//! and everything downstream sees an ordinary tensor. `brush-render`'s
//! `burn_glue.rs` does the same thing for the splat rasteriser's seven
//! outputs (its `BindOp`); this is the one-output, no-input case, kept
//! generic over the dtype so the `u32` buffers can cross too.
//!
//! Invariants:
//!   - `D` must equal the buffer's rank and the dtype must match; both are
//!     asserted, because a silent shape mismatch here does not fail — it
//!     surfaces much later as a wrong convolution.
//!   - The buffer must be contiguous. Not checked: every buffer the pyramid
//!     kernels produce comes straight from `create_tensor`/`create_tensor_from_slice`
//!     and so always is, and a `CubeTensor` carries its strides, so a
//!     hypothetical strided input would be handled correctly by the ops that
//!     read it rather than silently mis-read.
//!   - The bridge is **zero-copy**: no readback, no host allocation. The only
//!     cost is one stream registration.
//! Related docs: `rust/README.md`; `docs/LIMITATIONS.md`;
//!   `rust/brush-trips/crates/brush-render/src/burn_glue.rs`.

use brush_cube::MainBackendBase;
use burn::backend::{BackendTensor, DispatchTensor, DispatchTensorKind, TensorMetadata};
use burn::tensor::{DType, Int, Tensor};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::stream::{Operation, StreamId};
use burn_fusion::{get_client, FusionHandle};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::{CubeTensor, WgpuRuntime};

/// Whether a bound buffer is registered as a float or an int tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Float,
    Int,
}

/// The zero-input, one-output custom op that binds a concrete `CubeTensor`
/// into a fusion stream. See the module docs for why this indirection exists.
#[derive(Debug)]
struct BindOp {
    desc: CustomOpIr,
    tensor: CubeTensor<WgpuRuntime>,
    kind: Kind,
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOp {
    fn execute(&self, handles: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
        let (_, outputs) = self.desc.as_fixed::<0, 1>();
        let [out] = outputs;
        match self.kind {
            Kind::Float => handles.register_float_tensor::<MainBackendBase>(&out.id, self.tensor.clone()),
            Kind::Int => handles.register_int_tensor::<MainBackendBase>(&out.id, self.tensor.clone()),
        }
    }
}

/// Register `tensor` on the fusion stream of its own device and hand back the
/// resulting fusion primitive as a `DispatchTensor`.
///
/// `Wgpu = Fusion<CubeBackend<WgpuRuntime>>` in this Burn revision, so the
/// client to register on is the fusion client of the *inner* backend,
/// `MainBackendBase` (`burn-wgpu/src/lib.rs:35,76`).
fn bind(tensor: CubeTensor<WgpuRuntime>, kind: Kind) -> DispatchTensor {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let device = tensor.device();
    let client = get_client::<MainBackendBase>(&device);
    let out_ir = TensorIr::uninit(client.create_empty_handle(), shape, dtype);
    let desc = CustomOpIr::new("brush_pyramid_bind", &[], &[out_ir]);
    let op = BindOp {
        desc: desc.clone(),
        tensor,
        kind,
    };
    let fusion = client
        .register(StreamId::current(), OperationIr::Custom(desc), op)
        .output();
    let backend_tensor = match kind {
        Kind::Float => BackendTensor::Float(fusion),
        Kind::Int => BackendTensor::Int(fusion),
    };
    DispatchTensor {
        kind: DispatchTensorKind::Wgpu(backend_tensor),
        checkpointing: None,
    }
}

/// Wrap a device-resident f32 `CubeTensor` as a `burn::Tensor<D>`, zero-copy.
///
/// # Arguments
/// - `tensor`: an `f32` buffer of rank `D`, as produced by the pyramid
///   kernels. The fusion client is looked up from the tensor's own device.
///
/// # Panics
/// Panics if the buffer's rank is not `D` or its dtype is not `f32` — both
/// are programming errors at the call site, not runtime conditions.
#[must_use]
pub fn float_tensor<const D: usize>(tensor: CubeTensor<WgpuRuntime>) -> Tensor<D> {
    assert_eq!(
        tensor.shape().num_dims(),
        D,
        "float_tensor::<{D}> got a rank-{} buffer",
        tensor.shape().num_dims()
    );
    assert_eq!(tensor.dtype(), DType::F32, "float_tensor expects an f32 buffer");
    Tensor::from_dispatch(bind(tensor, Kind::Float))
}

/// Wrap a device-resident u32 `CubeTensor` as a `burn::Tensor<D, Int>`.
///
/// # Panics
/// Panics if the buffer's rank is not `D`.
#[must_use]
pub fn int_tensor<const D: usize>(tensor: CubeTensor<WgpuRuntime>) -> Tensor<D, Int> {
    assert_eq!(
        tensor.shape().num_dims(),
        D,
        "int_tensor::<{D}> got a rank-{} buffer",
        tensor.shape().num_dims()
    );
    Tensor::from_dispatch(bind(tensor, Kind::Int))
}
