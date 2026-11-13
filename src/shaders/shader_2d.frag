#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    if (fragUV.x < 0.0) {
        outColor = fragColor;
    } else {
        vec4 texColor = texture(texSampler, fragUV);
        outColor = fragColor * texColor;
    }
}
