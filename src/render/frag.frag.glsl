#version 450
precision highp float;

uniform vec2 r;     
uniform float t;    
out vec4 o;        

void main() {
    vec3 FC = vec3(gl_FragCoord.xy, 0.5); 
    vec3 rayDir = normalize(FC * 2.0 - r.xyy); 

    vec4 colorAccum = vec4(0.0);
    float z = 0.0;
    float d = 0.0;

    for (float i = 0.0; i < 80.0; i++) {
        vec3 p = z * rayDir;
        p.z += 6.0; // move camera back

        vec3 a = normalize(cos(vec3(1.0, 2.0, 0.0) + t - d * 5.0));//decides turbulence

        a = a * dot(a, p) - cross(a, p);

        for (d = 1.0; d <= 2.0; d++) {
            a += sin(a * d + t).yzx / d;
        }

        d = 0.05 * abs(length(p) - 3.0) + 0.04 * abs(a.y);
        d = max(d, 1e-4); // safety against divide-by-zero
        z += d;

        vec4 col = (cos(d / 0.1 + vec4(1.0, z, z, 0.0)) + 1.0); //decides color
        colorAccum += col / d * z;
    }

    o = vec4(tanh(colorAccum.rgb / 3e4), 1.0);
}
