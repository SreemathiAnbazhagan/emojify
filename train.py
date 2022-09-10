{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/SreemathiAnbazhagan/emojify/blob/main/train.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "lcDGhZfJgo9N"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import cv2\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Dropout, Flatten\n",
        "from tensorflow.keras.layers import Conv2D\n",
        "from tensorflow.keras.optimizers import Adam\n",
        "from tensorflow.keras.layers import MaxPooling2D\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CKCj-_ICihne"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9bUJItP4inJ-"
      },
      "source": [
        "#traning and validation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "magLct5si2L5",
        "outputId": "eb66b904-acbb-4ab9-e9b9-aa17ceef8817"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 28719 images belonging to 7 classes.\n",
            "Found 7178 images belonging to 7 classes.\n"
          ]
        }
      ],
      "source": [
        "train_dir = '/content/drive/MyDrive/Colab Notebooks/data/train'\n",
        "val_dir = '/content/drive/MyDrive/Colab Notebooks/data/test' \n",
        "train_datagen = ImageDataGenerator(rescale=1./255)\n",
        "val_datagen = ImageDataGenerator(rescale=1./255)\n",
        "train_generator = train_datagen.flow_from_directory(\n",
        "        train_dir,\n",
        "        target_size=(48,48),\n",
        "        batch_size=64,\n",
        "        color_mode=\"grayscale\",\n",
        "        class_mode='categorical')\n",
        "\n",
        "validation_generator = val_datagen.flow_from_directory(\n",
        "    val_dir,\n",
        "    target_size=(48,48),\n",
        "    batch_size=64,\n",
        "    color_mode=\"grayscale\",\n",
        "    class_mode='categorical')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TyLBuesdj-wu",
        "outputId": "13779507-324e-4dab-948d-a841889f5db9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/adam.py:105: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.\n",
            "  super(Adam, self).__init__(name, **kwargs)\n"
          ]
        }
      ],
      "source": [
        "emotion_model = Sequential()\n",
        "emotion_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48,48,1)))\n",
        "emotion_model.add(Conv2D(64,(3, 3), activation='relu'))\n",
        "emotion_model.add(MaxPooling2D(pool_size=(2,2)))\n",
        "emotion_model.add(Dropout(0.25))\n",
        "emotion_model.add(Conv2D(128, (3, 3), activation='relu'))\n",
        "emotion_model.add(MaxPooling2D(pool_size=(2,2)))\n",
        "emotion_model.add(Dropout(0.25))\n",
        "emotion_model.add(Flatten())\n",
        "emotion_model.add(Dense(1024, activation='relu'))\n",
        "emotion_model.add(Dropout(0.5))\n",
        "emotion_model.add(Dense(7, activation='softmax'))\n",
        "emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "BoYYMDMVkJm3",
        "outputId": "df92295a-854d-4a7a-c263-32cff52ad5ec"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:6: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
            "  \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/50\n",
            " 29/448 [>.............................] - ETA: 2:44:27 - loss: 1.8427 - accuracy: 0.2408"
          ]
        }
      ],
      "source": [
        "emotion_model_info = emotion_model.fit_generator(\n",
        "        train_generator,\n",
        "        steps_per_epoch=28709 // 64,\n",
        "        epochs=50,\n",
        "        validation_data=validation_generator,\n",
        "        validation_steps=7178 // 64)\n",
        "\n",
        "emotion_model_info = emotion_model.fit_generator(\n",
        "       train_generator,\n",
        "       steps_per_epoch=28709 //64,\n",
        "       epoch=50,\n",
        "       validation_data=validation_generator,\n",
        "       validation_steps=7178 //64)\n",
        "\n",
        "emotion_model.save_weights('model.hs')\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1ijYkIeXk2w6ykUL-SzVrSD3DfzZQCFdZ",
      "authorship_tag": "ABX9TyO91n800qEcavi8jdtElzzD",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}