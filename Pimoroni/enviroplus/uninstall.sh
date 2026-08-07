printf "It's recommended you run these steps manually.\n"
printf "If you want to run the full script, open it in\n"
printf "an editor and remove 'exit 1' from below.\n"
exit 1
source /home/team5/Team5/bin/activate
apt uninstall -y python3-cffi libportaudio2
python -m pip uninstall enviroplus
cp /home/team5/Pimoroni/config-backups/config.preinstall-enviroplus-2026-08-07-17-08-06.txt /boot/firmware/config.txt
rm -r /home/team5/Pimoroni/enviroplus
