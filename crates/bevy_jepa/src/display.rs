use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};
use bevy_burn::{BevyBurnHandle, BindingDirection, TransferKind};
use burn::tensor::Tensor;

use crate::{BevyJepaDisplayTransfer, JepaBevyBackend};

#[derive(Resource, Clone)]
pub(crate) struct JepaPanelTextures {
    pub(crate) root_entity: Option<Entity>,
    pub(crate) input_image: Handle<Image>,
    pub(crate) mask_image: Handle<Image>,
    pub(crate) low_res_image: Handle<Image>,
    pub(crate) high_res_image: Handle<Image>,
    pub(crate) input_entity: Option<Entity>,
    pub(crate) mask_entity: Option<Entity>,
    pub(crate) low_res_entity: Option<Entity>,
    pub(crate) high_res_entity: Option<Entity>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Default for JepaPanelTextures {
    fn default() -> Self {
        Self {
            root_entity: None,
            input_image: Handle::default(),
            mask_image: Handle::default(),
            low_res_image: Handle::default(),
            high_res_image: Handle::default(),
            input_entity: None,
            mask_entity: None,
            low_res_entity: None,
            high_res_entity: None,
            width: 1,
            height: 1,
        }
    }
}

#[derive(Component)]
pub(crate) struct OneShotGpuUpload;

pub(crate) enum InputPanelData {
    Tensor {
        width: u32,
        height: u32,
        input_rgba: Tensor<JepaBevyBackend, 3>,
    },
    Host {
        width: u32,
        height: u32,
        input_rgba: Vec<u8>,
    },
}

pub(crate) enum StagePanelData {
    Tensor {
        width: u32,
        height: u32,
        mask_rgba: Tensor<JepaBevyBackend, 3>,
        low_res_rgba: Tensor<JepaBevyBackend, 3>,
        high_res_rgba: Option<Tensor<JepaBevyBackend, 3>>,
    },
    Host {
        width: u32,
        height: u32,
        mask_rgba: Vec<u8>,
        low_res_rgba: Vec<u8>,
        high_res_rgba: Option<Vec<u8>>,
    },
}

pub(crate) fn apply_input_panel_to_world(
    world: &mut World,
    image_data: InputPanelData,
    transfer: BevyJepaDisplayTransfer,
) {
    let Some(texture) = world.get_resource::<JepaPanelTextures>().cloned() else {
        return;
    };
    let mut panel_size = (texture.width, texture.height);
    match (image_data, transfer) {
        (
            InputPanelData::Tensor {
                width,
                height,
                input_rgba,
            },
            BevyJepaDisplayTransfer::Gpu,
        ) => {
            panel_size = (width, height);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                set_gpu_visualization_image(&texture.input_image, width, height, &mut images);
            }
            set_gpu_panel_upload_handle(
                world,
                texture.input_entity,
                texture.input_image.clone(),
                input_rgba,
            );
        }
        (
            InputPanelData::Host {
                width,
                height,
                input_rgba,
            },
            _,
        ) => {
            panel_size = (width, height);
            remove_gpu_handle(world, texture.input_entity);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                set_host_visualization_image(
                    &texture.input_image,
                    width,
                    height,
                    input_rgba,
                    &mut images,
                );
            }
        }
        (InputPanelData::Tensor { .. }, BevyJepaDisplayTransfer::Cpu) => {}
    }
    if let Some(mut texture) = world.get_resource_mut::<JepaPanelTextures>() {
        texture.width = panel_size.0.max(1);
        texture.height = panel_size.1.max(1);
    }
}

pub(crate) fn apply_stage_panels_to_world(
    world: &mut World,
    image_data: StagePanelData,
    transfer: BevyJepaDisplayTransfer,
) {
    let Some(texture) = world.get_resource::<JepaPanelTextures>().cloned() else {
        return;
    };
    let mut panel_size = (texture.width, texture.height);
    match (image_data, transfer) {
        (
            StagePanelData::Tensor {
                width,
                height,
                mask_rgba,
                low_res_rgba,
                high_res_rgba,
            },
            BevyJepaDisplayTransfer::Gpu,
        ) => {
            panel_size = (width, height);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                for handle in [&texture.mask_image, &texture.low_res_image] {
                    set_gpu_visualization_image(handle, width, height, &mut images);
                }
                if high_res_rgba.is_some() {
                    set_gpu_visualization_image(
                        &texture.high_res_image,
                        width,
                        height,
                        &mut images,
                    );
                }
            }
            set_gpu_panel_upload_handle(
                world,
                texture.mask_entity,
                texture.mask_image.clone(),
                mask_rgba,
            );
            set_gpu_panel_upload_handle(
                world,
                texture.low_res_entity,
                texture.low_res_image.clone(),
                low_res_rgba,
            );
            if let Some(high_res_rgba) = high_res_rgba {
                set_gpu_panel_upload_handle(
                    world,
                    texture.high_res_entity,
                    texture.high_res_image.clone(),
                    high_res_rgba,
                );
            }
        }
        (
            StagePanelData::Host {
                width,
                height,
                mask_rgba,
                low_res_rgba,
                high_res_rgba,
            },
            _,
        ) => {
            panel_size = (width, height);
            remove_panel_gpu_handles(world, &texture);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                set_host_visualization_image(
                    &texture.mask_image,
                    width,
                    height,
                    mask_rgba,
                    &mut images,
                );
                set_host_visualization_image(
                    &texture.low_res_image,
                    width,
                    height,
                    low_res_rgba,
                    &mut images,
                );
                if let Some(high_res_rgba) = high_res_rgba {
                    set_host_visualization_image(
                        &texture.high_res_image,
                        width,
                        height,
                        high_res_rgba,
                        &mut images,
                    );
                }
            }
        }
        (StagePanelData::Tensor { .. }, BevyJepaDisplayTransfer::Cpu) => {}
    }
    if let Some(mut texture) = world.get_resource_mut::<JepaPanelTextures>() {
        texture.width = panel_size.0.max(1);
        texture.height = panel_size.1.max(1);
    }
}

pub(crate) fn clear_completed_gpu_uploads(
    mut commands: Commands,
    mut query: Query<(Entity, &mut BevyBurnHandle<JepaBevyBackend>), With<OneShotGpuUpload>>,
) {
    for (entity, handle) in &mut query {
        if handle.upload {
            continue;
        }
        commands.entity(entity).remove::<OneShotGpuUpload>();
    }
}

fn set_gpu_panel_upload_handle(
    world: &mut World,
    entity: Option<Entity>,
    image: Handle<Image>,
    tensor: Tensor<JepaBevyBackend, 3>,
) {
    let Some(entity) = entity else {
        return;
    };
    let Ok(mut entity) = world.get_entity_mut(entity) else {
        return;
    };
    if let Some(mut handle) = entity.get_mut::<BevyBurnHandle<JepaBevyBackend>>() {
        handle.bevy_image = image;
        handle.tensor = tensor;
        handle.direction = BindingDirection::BurnToBevy;
        handle.xfer = TransferKind::Gpu;
        handle.upload = true;
    } else {
        entity.insert(BevyBurnHandle::<JepaBevyBackend> {
            bevy_image: image,
            tensor,
            upload: true,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Gpu,
        });
    }
    entity.insert(OneShotGpuUpload);
}

fn remove_panel_gpu_handles(world: &mut World, texture: &JepaPanelTextures) {
    for entity in [
        texture.input_entity,
        texture.mask_entity,
        texture.low_res_entity,
        texture.high_res_entity,
    ]
    .into_iter()
    .flatten()
    {
        remove_gpu_handle(world, Some(entity));
    }
}

fn remove_gpu_handle(world: &mut World, entity: Option<Entity>) {
    if let Some(entity) = entity
        && let Ok(mut entity) = world.get_entity_mut(entity)
    {
        entity.remove::<BevyBurnHandle<JepaBevyBackend>>();
    }
}

fn set_gpu_visualization_image(
    handle: &Handle<Image>,
    width: u32,
    height: u32,
    images: &mut Assets<Image>,
) {
    let width = width.max(1);
    let height = height.max(1);
    if let Some(image) = images.get(handle)
        && image.width() == width
        && image.height() == height
        && image.texture_descriptor.format == TextureFormat::Rgba32Float
        && image.texture_descriptor.usage.contains(
            TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
        )
    {
        return;
    }
    let _ = images.insert(handle.id(), gpu_visualization_image(width, height));
}

fn gpu_visualization_image(width: u32, height: u32) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING;
    image.sampler = ImageSampler::nearest();
    image
}

fn set_host_visualization_image(
    handle: &Handle<Image>,
    width: u32,
    height: u32,
    mut rgba: Vec<u8>,
    images: &mut Assets<Image>,
) {
    let width = width.max(1);
    let height = height.max(1);
    let expected_len = width as usize * height as usize * 4;
    if rgba.len() != expected_len {
        rgba.resize(expected_len, 0);
    }

    if let Some(mut image) = images.get_mut(handle)
        && image.width() == width
        && image.height() == height
        && image.texture_descriptor.format == TextureFormat::Rgba8UnormSrgb
    {
        image.data = Some(rgba);
        return;
    }
    let _ = images.insert(handle.id(), host_visualization_image(width, height, rgba));
}

fn host_visualization_image(width: u32, height: u32, rgba: Vec<u8>) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::nearest();
    image
}
