---
hide_title: true
sidebar_label: Installation of jetpack 4.3 & 4.6
---
## For quick installaton of jetpack 4.3 & 4.6 on tx2 with Auvidea J120 board
### Instruction for running the sh file:
  ``` bash
  chmod +x <name_of_sh_file>.sh
  ./<name_of_sh_file>.sh
  #Make sure to put tx2 into force recovery mode.
  ```

:::caution 
(For Jetpack 4.6) Make sure to use Auvidea J120 Motherboard that are manufactured later than 08/2018 if not it will not work with this script. Another alternative will be J121 (03/2019) Auvidea motherboard.
:::

### Copy all steps inside and run this sh file for jetpack 4.3:
-Copy the below codes into a text file and save as a sh file for ease of installation. 
  ``` bash
  #! /bin/sh
  #filename: jetpack_4.3.sh

  echo -e "$(tput setaf 185) \n Downloading Auvidea Firmware version 3.0, supporting Jetpack 4.3 $(tput sgr 0) \n "
  cd ~/Downloads
  wget -O J90-J120-J130_4_3.tar.bz2 https://www.dropbox.com/s/m8uub4aaiyv66vw/J90-J120-J130_4_3.tar.bz2

  echo -e "$(tput setaf 185) \n Downloading Jetpack 4.3 \n $(tput sgr 0) "
  wget -O Tegra186_Linux_R32.3.1_aarch64.tbz2 https://www.dropbox.com/s/2bytjd4cato9keo/     Tegra186_Linux_R32.3.1_aarch64.tbz2

  echo -e "$(tput setaf 185) \n Downloading jetpack 4.3 sample rootfs folder $(tput sgr 0) \n "
  wget -O Tegra_Linux_Sample-Root-Filesystem_R32.3.1_aarch64.tbz2 https://www.dropbox.com/s/ ec64faryyos44sr/Tegra_Linux_Sample-Root-Filesystem_R32.3.1_aarch64.tbz2

  echo -e "$(tput setaf 185) \n Extracting L4T BSP driver to home $(tput sgr 0)\n"
  cd ~/
  sudo tar -xpf ~/Downloads/Tegra186_Linux_R32.3.1_aarch64.tbz2
  sudo mv Linux_for_Tegra/ Linux_for_Tegra_4.3

  echo -e "$(tput setaf 185) \n Extracting sample file to rootfs folder $(tput sgr 0)\n"
  cd ~/Linux_for_Tegra_4.3/rootfs
  sudo tar -xpf ~/Downloads/Tegra_Linux_Sample-Root-Filesystem_R32.3.1_aarch64.tbz2

  echo -e "$(tput setaf 185) \n Extracting and copying auvidea kernel file and updating      Linux_for_Tegra_4.3 file $(tput sgr 0)\n"
  cd ~/Downloads
  sudo tar -xpf J90-J120-J130_4_3.tar.bz2
  cd ~/Downloads/J90-J120-J130_4_3
  sudo tar -xpf kernel_out.tar.bz2
  sudo cp -a Linux_for_Tegra/* ~/Linux_for_Tegra_4.3/

  echo -e "$(tput setaf 185) \n Applying Binaries to Linux_for_Tegra $(tput sgr 0)\n"
  cd ~/
  sudo apt-get install python3
  sudo apt-get install qemu-user-static -y
  sudo ln -s /usr/bin/python3 /usr/bin/python
  cd ~/Linux_for_Tegra_4.3
  sudo ./apply_binaries.sh

  echo -e "$(tput setaf 185) \n Removing downloaded files $(tput sgr 0)\n"
  cd ~/Downloads
  sudo rm -rf ./J90-J120-J130_4_3.tar.bz2
  sudo rm -rf ./Tegra186_Linux_R32.3.1_aarch64.tbz2
  sudo rm -rf ./Tegra_Linux_Sample-Root-Filesystem_R32.3.1_aarch64.tbz2
  sudo rm -rf ./J90-J120-J130_4_3

  echo -e "$(tput setaf 185) \n You can now start to flash on jetson TX2 $(tput sgr 0)\n"
  ```
### Copy all steps inside and run this sh file for jetpack 4.6:
-Copy the below codes into a text file and save as a sh file for ease of installation.
  ``` bash
  #! /bin/sh
  #filename: jetpack_4.6.sh

  echo -e "$(tput setaf 185) \n Downloading Auvidea Firmware, supporting Jetpack 4.6 $(tput sgr 0) \n "
  cd ~/Downloads
  wget -O kernel_out.tar.bz2 https://www.dropbox.com/s/9i6d27y8zrr0gsv/kernel_out.tar.bz2

  echo -e "$(tput setaf 185) \n Downloading Jetpack 4.6 \n $(tput sgr 0) "
  wget -O Jetson_Linux_R32.6.1_aarch64.tbz2 https://www.dropbox.com/s/hirds357pezpnxb/  Jetson_Linux_R32.6.1_aarch64.tbz2

  echo -e "$(tput setaf 185) \n Downloading jetpack 4.6 sample rootfs folder $(tput sgr 0) \n "
  wget -O Tegra_Linux_Sample-Root-Filesystem_R32.6.1_aarch64.tbz2 https://www.dropbox.com/s/  eqjltpvb35talf2/Tegra_Linux_Sample-Root-Filesystem_R32.6.1_aarch64.tbz2

  echo -e "$(tput setaf 185) \n Extracting L4T BSP driver to home $(tput sgr 0)\n"
  cd ~/
  sudo tar -xpf ~/Downloads/Jetson_Linux_R32.6.1_aarch64.tbz2
  sudo mv Linux_for_Tegra/ Linux_for_Tegra_4.6 #edit here for raw

  echo -e "$(tput setaf 185) \n Extracting sample file to rootfs folder $(tput sgr 0)\n"
  cd ~/Linux_for_Tegra_4.6/rootfs #edit here for raw
  sudo tar -xpf ~/Downloads/Tegra_Linux_Sample-Root-Filesystem_R32.6.1_aarch64.tbz2

  echo -e "$(tput setaf 185) \n Extracting and copying auvidea kernel file and updating   Linux_for_Tegra_4.6 file $(tput sgr 0)\n"
  cd ~/Downloads
  sudo tar -xpf kernel_out.tar.bz2
  sudo cp -a kernel_out/* ~/Linux_for_Tegra_4.6/ #edit here for raw

  echo -e "$(tput setaf 185) \n Applying Binaries to Linux_for_Tegra $(tput sgr 0)\n"
  cd ~/
  sudo apt-get install python3
  sudo apt-get install qemu-user-static -y
  sudo ln -s /usr/bin/python3 /usr/bin/python
  cd ~/Linux_for_Tegra_4.6 #edit here for raw
  sudo ./apply_binaries.sh

  echo -e "$(tput setaf 185) \n Removing downloaded files $(tput sgr 0)\n"
  cd ~/Downloads
  sudo rm -rf ./kernel_out.tar.bz2
  sudo rm -rf ./Jetson_Linux_R32.6.1_aarch64.tbz2
  sudo rm -rf ./Tegra_Linux_Sample-Root-Filesystem_R32.6.1_aarch64.tbz2
  sudo rm -rf ./kernel_out

  echo -e "$(tput setaf 185) \n You can now start to flash on jetson TX2 $(tput sgr 0)\n"
  ```

### Flashing of the raw system img (Jetpack 4.3 & 4.6):
- Make sure to cd to the correct folder.
  ``` bash
  cd ~/Linux_for_Tegra_4.3
  #cd ~/Linux_for_Tegra_4.6
  sudo ./flash.sh jetson-tx2 mmcblk0p1
  ```

### Copying a older tx2 image and flashng on other tx2:
  ``` bash 
  cd ~/Linux_for_Tegra_4.3
  #cd ~/Linux_for_Tegra_4.6
  sudo ./flash.sh -r -k APP -G backup.img jetson-tx2 mmcblk0p1   #Only need to do once
  sudo cp backup.img.raw bootloader/system.img                   #Only need to do once
  sudo ./flash.sh -r jetson-tx2 mmcblk0p1
  ```



