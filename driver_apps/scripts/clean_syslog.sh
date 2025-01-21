#!/bin/bash -xe

sudo sed -i '/f,/d; /readp/d; /hpcs/d; /b,/d; /s,/d' /var/log/syslog
