#version 450
out vec3 FC;
        
uniform vec2 r; // resolution
        
void main() {
  vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  FC = vec3(uv, 0.0); // add .z for use in FC.rgb
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
