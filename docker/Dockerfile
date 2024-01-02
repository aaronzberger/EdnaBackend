FROM ubuntu:jammy

# set environment variables for tzdata
ARG TZ=America/New_York
ENV TZ=${TZ}

# include manual pages and documentation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&\
    yes | unminimize

# install interactive programs (emacs, vim, nano, man, sudo, etc.)
RUN apt-get -y install\
    bc\
    curl\
    dc\
    git\
    git-doc\
    man\
    micro\
    nano\
    psmisc\
    sudo\
    vim\
    wget\
    locales\
    python3\
    python3-pip

RUN pip install --no-cache-dir\
    jsonpickle\
    scikit-learn-extra\
    branca\
    certifi\
    charset-normalizer\
    cycler\
    folium\
    fonttools\
    geographiclib\
    geojson\
    haversine\
    rapidfuzz\
    idna\
    Jinja2\
    kiwisolver\
    MarkupSafe\
    matplotlib\
    mccabe\
    nptyping\
    numpy\
    overpass\
    packaging\
    Pillow\
    polyline\
    pycodestyle\
    pyflakes\
    pyparsing\
    python-dateutil\
    requests\
    Shapely\
    six\
    termcolor\
    tqdm\
    urllib3\
    utm\
    humanhash3\
    names\
    redis[hiredis]\
    typing_extensions\
    ortools\
    pandas\
    boto3\
    python-dotenv

# set up default locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

# remove unneeded .deb files
RUN rm -r /var/lib/apt/lists/*

# set up passwordless sudo for user user
RUN useradd -m -s /bin/bash user && \
    echo "user ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/init

# create binary reporting version of dockerfile
RUN (echo '#\!/bin/sh'; echo 'echo 16') > /usr/bin/docker-version; chmod ugo+rx,u+w,go-w /usr/bin/docker-version

# git build arguments
ARG USER=User
ARG EMAIL=nobody@example.com

# configure your environment
USER user
RUN git config --global user.name "${USER}" && \
    git config --global user.email "${EMAIL}" && \
    (echo "(custom-set-variables"; echo " '(c-basic-offset 4)"; echo " '(indent-tabs-mode nil))") > ~/.emacs && \
    (echo "set expandtab"; echo "set shiftwidth=4"; echo "set softtabstop=4") > ~/.vimrc && \
    cat /dev/null > ~/.bash_profile && \
    echo "# 2022: avoid a Docker bug with user mapping by listing working directory" >> ~/.bash_profile && \
    echo "ls -al > /dev/null" >> ~/.bash_profile && \
    echo "for i in \`mount | grep /home/user | sed 's/^.*\\(\\/home[^ ]*\\).*/\\\\1/'\`; do ls -al \$i > /dev/null; done" >> ~/.bash_profile && \
    echo "# make ssh-auth.sock user-readable" >> ~/.bash_profile && \
    (echo "if test -e /run/host-services/ssh-auth.sock; then"; echo "  sudo chown user:user /run/host-services/ssh-auth.sock"; echo "fi") >> ~/.bash_profile && \
    echo ". ~/.bashrc" >> ~/.bash_profile && \
    rm -f ~/.bash_logout && \
    echo "add-auto-load-safe-path ~" > ~/.gdbinit

WORKDIR /home/user
CMD ["/bin/bash", "-l"]

# Install Rust, Cargo, and the VRP solver
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/home/user/.cargo/bin:${PATH}"
RUN cargo install vrp-cli