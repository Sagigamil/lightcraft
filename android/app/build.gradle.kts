plugins { id("com.android.application") }
android {
    namespace = "com.lightcraft.android"
    compileSdk = 35
    ndkVersion = "28.2.13676358"
    defaultConfig {
        applicationId = "com.lightcraft.android"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.4.0-android"
        ndk { abiFilters += providers.gradleProperty("androidAbi").getOrElse("arm64-v8a") }
    }
    buildTypes {
        getByName("release") {
            // Local sideload build only. Production keys are never stored in this repo.
            signingConfig = signingConfigs.getByName("debug")
            isMinifyEnabled = false
        }
    }
    flavorDimensions += "edition"
    productFlavors {
        // Lite is the existing application: no AI weights, libraries, assets or menu entries.
        create("lite") {
            dimension = "edition"
            applicationId = "com.lightcraft.android"
            versionNameSuffix = ""
        }
        // AI embeds the point-prompt SAM 3 checkpoint and the pure-Rust inference runtime.
        create("ai") {
            dimension = "edition"
            applicationId = "com.lightcraft.android.ai"
            versionNameSuffix = "-ai"
        }
    }
    androidResources {
        // The F16 checkpoint is incompressible; store it verbatim so aapt2 does not
        // spend minutes compressing ~950 MB (and so extraction is a straight copy).
        noCompress += listOf("safetensors")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging { jniLibs { useLegacyPackaging = false } }
}
