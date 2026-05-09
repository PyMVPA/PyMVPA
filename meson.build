project(
    'pymvpa2',
    version : '2.6.5.dev1',
    license : 'MIT',
    meson_version : '>=1.1.0',
    default_options : [
        'warning_level=1',
    ],
)

# Phase 6a: pure-Python install only.  Phase 6b will add 'c' / 'cpp' to
# project languages and bring in the libsvmc / libsmlrc extensions.
py = import('python').find_installation(pure : true)

install_subdir(
    'mvpa2',
    install_dir : py.get_install_dir(),
    install_tag : 'python-runtime',
    exclude_directories : [
        '__pycache__',
    ],
    exclude_files : [
        # Stale SWIG-generated artifacts re-introduced in Phase 6b.
        'clfs/libsvmc/svmc.py',
        'clfs/libsvmc/svmc_wrap.cpp',
    ],
)

# bin/ scripts (full Python files with shebangs, not entry-point strings)
install_data(
    'bin/pymvpa2',
    'bin/pymvpa2-prep-afni-surf',
    'bin/pymvpa2-prep-fmri',
    'bin/pymvpa2-tutorial',
    install_dir : get_option('bindir'),
    install_mode : 'rwxr-xr-x',
)
