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
        // SAM 3 embeds the point-prompt checkpoint and the chained QNN contexts.
        create("sam3") {
            dimension = "edition"
            applicationId = "com.lightcraft.android"
            versionNameSuffix = "-sam3"
        }
        // MobileSAM embeds the HTP encoder + CPU decoder QNN contexts.
        create("mobilesam") {
            dimension = "edition"
            applicationId = "com.lightcraft.android"
            versionNameSuffix = "-mobilesam"
        }
    }
    androidResources {
        // The F16 checkpoint and the QNN context binaries are incompressible; store them
        // verbatim so aapt2 does not spend minutes compressing ~2 GB (and so extraction is
        // a straight copy).
        noCompress += listOf("safetensors", "bin", "so")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging { jniLibs { useLegacyPackaging = true } }
}
