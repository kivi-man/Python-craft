// PythonCraft Engine - Fragment Shader (Vanilla Minecraft Style)
#version 330 core

in vec3 vColor;
in vec3 vNormal;
in vec3 vFragPos;
in vec3 vTexCoord;
in float vAO;
in float vLight;
in float vOverlay;

uniform sampler2DArray u_texture;
uniform vec4 u_tint_color = vec4(1.0);

out vec4 FragColor;

void main() {
    // 1. Orijinal Minecraft Yüzey Işıklandırma Çarpanları
    float light_factor = 1.0;
    vec3 n = normalize(vNormal);
    
    // Yüzey yönüne göre gölgeleme (Vanilla Minecraft Değerleri)
    if (n.y > 0.5) {
        light_factor = 1.0;         // Üst yüzeyler (Güneş tam vurur)
    } else if (n.y < -0.5) {
        light_factor = 0.5;         // Alt yüzeyler (Tam gölge)
    } else if (abs(n.z) > 0.5) {
        light_factor = 0.8;         // Kuzey/Güney yüzeyleri (Kısmi gölge)
    } else if (abs(n.x) > 0.5) {
        light_factor = 0.6;         // Doğu/Batı yüzeyleri (Daha koyu gölge)
    }
    
    // 2. Ambient Occlusion Çarpanı (0=Karanlık, 3=Aydınlık)
    // Minecraft AO değerleri: 0 -> %50, 1 -> %70, 2 -> %85, 3 -> %100
    // Orijinal Minecraft'ta AO daha serttir. Biz de sertleştirdik.
    float ao_multiplier = 0.5 + (vAO * 0.166); 
    
    // 3. SkyLight Çarpanı (Mağaralar için)
    float normalized_light = vLight / 15.0;
    float sky_light_factor = max(pow(normalized_light, 1.4), 0.05);
    
    // 4. Texture Sampling
    vec4 texColor = texture(u_texture, vTexCoord);
    
    // Alpha Test: effaf pikselleri zme (Derinlik tamponuna yazlmasn engeller)
    if (texColor.a < 0.1) {
        discard;
    }
    
    vec3 base_color;
    if (vOverlay > 0.0) {
        vec4 overlayColor = texture(u_texture, vec3(vTexCoord.xy, vOverlay));
        if (overlayColor.a > 0.0) {
            // Apply vColor (biome tint) ONLY to the overlay grass
            vec3 tintedOverlay = overlayColor.rgb * vColor;
            // Alpha blend overlay over base texture (dirt)
            base_color = mix(texColor.rgb, tintedOverlay, overlayColor.a);
        } else {
            base_color = texColor.rgb;
        }
    } else {
        // Normal textures
        base_color = vColor * texColor.rgb;
    }
    
    // Uygula Tint Rengini (Hasar efekti vs. iin)
    base_color *= u_tint_color.rgb;
    
    // Nihai Rengi Hesapla
    vec3 lit_color = base_color * light_factor * ao_multiplier * sky_light_factor;
    
    FragColor = vec4(lit_color, texColor.a);
}
