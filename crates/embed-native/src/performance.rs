//! CPU worker pool restricted to performance cores on heterogeneous systems.

use crate::{Error, Result};

pub struct PerformanceCorePool {
    inner: rayon::ThreadPool,
}

impl PerformanceCorePool {
    pub fn new(thread_prefix: &'static str) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let cpus = linux_performance_cpus()?;
            let worker_cpus = std::sync::Arc::new(cpus);
            let startup_cpus = worker_cpus.clone();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(worker_cpus.len())
                .thread_name(move |idx| format!("{thread_prefix}-pcore-{idx}"))
                .start_handler(move |idx| {
                    if let Some(cpu) = linux_cpu_for_worker(&startup_cpus, idx) {
                        let _ = linux_set_current_affinity(cpu);
                    }
                })
                .build()
                .map_err(|e| Error::Cpu(format!("cannot create performance-core pool: {e}")))?;
            if !pool
                .broadcast(|context| {
                    linux_cpu_for_worker(&worker_cpus, context.index())
                        .is_some_and(linux_set_current_affinity)
                })
                .into_iter()
                .all(|configured| configured)
            {
                return Err(Error::Cpu(
                    "cannot restrict CPU workers to Linux performance cores".into(),
                ));
            }
            return Ok(Self { inner: pool });
        }

        #[cfg(target_os = "macos")]
        {
            let threads = macos_performance_cpu_count().ok_or_else(|| {
                Error::Cpu("cannot determine Apple Silicon performance-core count".into())
            })?;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |idx| format!("{thread_prefix}-pcore-{idx}"))
                .start_handler(|_| {
                    let _ = macos_select_performance_qos();
                })
                .build()
                .map_err(|e| Error::Cpu(format!("cannot create performance-core pool: {e}")))?;
            if !pool
                .broadcast(|_| macos_select_performance_qos())
                .into_iter()
                .all(|configured| configured)
            {
                return Err(Error::Cpu(
                    "cannot assign performance QoS to CPU workers".into(),
                ));
            }
            return Ok(Self { inner: pool });
        }

        #[cfg(windows)]
        {
            let cpu_sets = windows_performance_cpu_sets()?;
            let worker_sets = std::sync::Arc::new(cpu_sets);
            let startup_sets = worker_sets.clone();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(worker_sets.len())
                .thread_name(move |idx| format!("{thread_prefix}-pcore-{idx}"))
                .start_handler(move |_| {
                    let _ = windows_set_current_cpu_sets(&startup_sets);
                })
                .build()
                .map_err(|e| Error::Cpu(format!("cannot create performance-core pool: {e}")))?;
            if !pool
                .broadcast(|_| windows_set_current_cpu_sets(&worker_sets))
                .into_iter()
                .all(|configured| configured)
            {
                return Err(Error::Cpu(
                    "cannot restrict CPU workers to Windows performance cores".into(),
                ));
            }
            return Ok(Self { inner: pool });
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
        {
            let threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |idx| format!("{thread_prefix}-cpu-{idx}"))
                .build()
                .map_err(|e| Error::Cpu(format!("cannot create CPU worker pool: {e}")))?;
            Ok(Self { inner: pool })
        }
    }

    pub fn install<OP, R>(&self, operation: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.inner.install(operation)
    }

    #[cfg(test)]
    fn thread_count(&self) -> usize {
        self.inner.current_num_threads()
    }
}

#[cfg(windows)]
fn windows_performance_cpu_sets() -> Result<Vec<u32>> {
    use windows_sys::Win32::System::SystemInformation::{
        CpuSetInformation, GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION,
    };

    let mut required = 0u32;
    unsafe {
        GetSystemCpuSetInformation(
            std::ptr::null_mut(),
            0,
            &mut required,
            std::ptr::null_mut(),
            0,
        );
    }
    if required == 0 {
        return Err(Error::Cpu(
            "Windows did not report any CPU-set information".into(),
        ));
    }
    let buffer_len = usize::try_from(required)
        .map_err(|_| Error::Cpu("Windows CPU-set buffer length does not fit usize".into()))?;
    let mut buffer = vec![0u8; buffer_len];
    if unsafe {
        GetSystemCpuSetInformation(
            buffer.as_mut_ptr().cast::<SYSTEM_CPU_SET_INFORMATION>(),
            required,
            &mut required,
            std::ptr::null_mut(),
            0,
        )
    } == 0
    {
        return Err(Error::Cpu(format!(
            "cannot enumerate Windows CPU sets: {}",
            std::io::Error::last_os_error()
        )));
    }

    let returned = usize::try_from(required)
        .map_err(|_| Error::Cpu("Windows CPU-set result length does not fit usize".into()))?
        .min(buffer.len());
    let header_size = std::mem::size_of::<u32>() + std::mem::size_of::<i32>();
    let mut offset = 0usize;
    let mut classified = Vec::new();
    while offset.saturating_add(header_size) <= returned {
        let info = unsafe {
            std::ptr::read_unaligned(
                buffer
                    .as_ptr()
                    .add(offset)
                    .cast::<SYSTEM_CPU_SET_INFORMATION>(),
            )
        };
        let size = usize::try_from(info.Size)
            .map_err(|_| Error::Cpu("Windows CPU-set entry size does not fit usize".into()))?;
        if size < header_size || offset.saturating_add(size) > returned {
            return Err(Error::Cpu(
                "Windows returned malformed CPU-set information".into(),
            ));
        }
        if info.Type == CpuSetInformation {
            let cpu_set = unsafe { info.Anonymous.CpuSet };
            classified.push((cpu_set.Id, cpu_set.EfficiencyClass));
        }
        offset = offset.saturating_add(size);
    }
    // Windows assigns larger EfficiencyClass values to faster, less
    // power-efficient cores.
    let best_class = classified
        .iter()
        .map(|(_, efficiency)| *efficiency)
        .max()
        .ok_or_else(|| Error::Cpu("Windows reported no usable CPU sets".into()))?;
    let selected = classified
        .into_iter()
        .filter_map(|(id, efficiency)| (efficiency == best_class).then_some(id))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        Err(Error::Cpu(
            "Windows reported no performance-class CPU sets".into(),
        ))
    } else {
        Ok(selected)
    }
}

#[cfg(windows)]
fn windows_set_current_cpu_sets(cpu_sets: &[u32]) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};

    let Ok(count) = u32::try_from(cpu_sets.len()) else {
        return false;
    };
    unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), cpu_sets.as_ptr(), count) != 0 }
}

#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Copy, Debug)]
struct LinuxCpuInfo {
    cpu: usize,
    core_type: Option<u64>,
    capacity: Option<u64>,
    max_frequency: Option<u64>,
    physical_package_id: Option<i64>,
    core_id: Option<i64>,
}

#[cfg(target_os = "linux")]
fn linux_performance_cpus() -> Result<Vec<usize>> {
    let mut allowed = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut allowed) }
        != 0
    {
        return Err(Error::Cpu(
            "cannot read Linux CPU affinity for native inference".into(),
        ));
    }

    let allowed = (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &allowed) })
        .collect::<Vec<_>>();
    if allowed.is_empty() {
        return Err(Error::Cpu(
            "Linux CPU affinity contains no processors".into(),
        ));
    }

    let mut discovered = std::fs::read_to_string("/sys/devices/system/cpu/online")
        .ok()
        .and_then(|online| parse_linux_cpu_list(&online).ok())
        .unwrap_or_else(|| allowed.clone());
    for &cpu in &allowed {
        if !discovered.contains(&cpu) {
            discovered.push(cpu);
        }
    }
    discovered.sort_unstable();
    discovered.dedup();

    let cpus = discovered
        .into_iter()
        .map(linux_cpu_info)
        .collect::<Vec<_>>();
    select_linux_performance_cpus(&cpus, &allowed)
}

#[cfg(target_os = "linux")]
fn linux_cpu_info(cpu: usize) -> LinuxCpuInfo {
    LinuxCpuInfo {
        cpu,
        core_type: linux_sysfs_cpu_value(cpu, "topology/core_type"),
        capacity: linux_sysfs_cpu_value(cpu, "cpu_capacity"),
        max_frequency: linux_sysfs_cpu_value(cpu, "cpufreq/cpuinfo_max_freq"),
        physical_package_id: linux_sysfs_cpu_value(cpu, "topology/physical_package_id"),
        core_id: linux_sysfs_cpu_value(cpu, "topology/core_id"),
    }
}

#[cfg(target_os = "linux")]
fn linux_sysfs_cpu_value<T>(cpu: usize, relative_path: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/{relative_path}");
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(any(target_os = "linux", test))]
fn select_linux_performance_cpus(cpus: &[LinuxCpuInfo], allowed: &[usize]) -> Result<Vec<usize>> {
    let performance = linux_performance_candidates(cpus)?;
    let allowed = allowed
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut performance = performance
        .into_iter()
        .filter(|cpu| allowed.contains(&cpu.cpu))
        .collect::<Vec<_>>();
    performance.sort_unstable_by_key(|cpu| cpu.cpu);

    let mut seen_cores = std::collections::HashSet::new();
    let mut selected = performance
        .into_iter()
        .filter(|cpu| match (cpu.physical_package_id, cpu.core_id) {
            (Some(package), Some(core)) if package >= 0 && core >= 0 => {
                seen_cores.insert((package, core))
            }
            _ => true,
        })
        .map(|cpu| cpu.cpu)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    if selected.is_empty() {
        Err(Error::Cpu(
            "Linux CPU affinity excludes every detected performance core".into(),
        ))
    } else {
        Ok(selected)
    }
}

#[cfg(any(target_os = "linux", test))]
fn linux_performance_candidates(cpus: &[LinuxCpuInfo]) -> Result<Vec<&LinuxCpuInfo>> {
    if cpus.is_empty() {
        return Err(Error::Cpu("Linux online CPU list is empty".into()));
    }

    let core_types = cpus
        .iter()
        .filter_map(|cpu| cpu.core_type)
        .collect::<std::collections::HashSet<_>>();
    if core_types.len() > 1 {
        let performance_type = core_types
            .into_iter()
            .max()
            .expect("multiple Linux core types cannot be empty");
        return Ok(cpus
            .iter()
            .filter(|cpu| cpu.core_type == Some(performance_type))
            .collect());
    }

    let complete_core_type = cpus.iter().all(|cpu| cpu.core_type.is_some());
    if complete_core_type {
        return Ok(cpus.iter().collect());
    }

    if let Some(performance) = linux_highest_scored_cpus(cpus, |cpu| cpu.capacity)
        .or_else(|| linux_highest_scored_cpus(cpus, |cpu| cpu.max_frequency))
    {
        return Ok(performance);
    }

    let any_classification_metadata = cpus.iter().any(|cpu| {
        cpu.core_type.is_some() || cpu.capacity.is_some() || cpu.max_frequency.is_some()
    });
    // Homogeneous hosts commonly expose no heterogeneous-core metadata. A
    // single complete core type is also homogeneous from the kernel's view.
    if complete_core_type || !any_classification_metadata {
        return Ok(cpus.iter().collect());
    }
    Err(Error::Cpu(
        "incomplete Linux performance-core metadata; refusing to schedule inference on unclassified processors"
            .into(),
    ))
}

#[cfg(any(target_os = "linux", test))]
fn linux_highest_scored_cpus(
    cpus: &[LinuxCpuInfo],
    score: fn(&LinuxCpuInfo) -> Option<u64>,
) -> Option<Vec<&LinuxCpuInfo>> {
    let scores = cpus.iter().filter_map(score).collect::<Vec<_>>();
    if scores.len() != cpus.len() {
        return None;
    }
    let max_score = scores.into_iter().max()?;
    Some(
        cpus.iter()
            .filter(|cpu| score(cpu) == Some(max_score))
            .collect(),
    )
}

#[cfg(target_os = "linux")]
fn parse_linux_cpu_list(value: &str) -> Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = part.split_once('-') {
            let start = start
                .parse::<usize>()
                .map_err(|_| Error::Cpu(format!("invalid Linux CPU range `{part}`")))?;
            let end = end
                .parse::<usize>()
                .map_err(|_| Error::Cpu(format!("invalid Linux CPU range `{part}`")))?;
            if end < start {
                return Err(Error::Cpu(format!("invalid Linux CPU range `{part}`")));
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(
                part.parse::<usize>()
                    .map_err(|_| Error::Cpu(format!("invalid Linux CPU id `{part}`")))?,
            );
        }
    }
    if cpus.is_empty() {
        return Err(Error::Cpu("Linux CPU list is empty".into()));
    }
    Ok(cpus)
}

#[cfg(any(target_os = "linux", test))]
fn linux_cpu_for_worker(cpus: &[usize], worker_index: usize) -> Option<usize> {
    cpus.get(worker_index).copied()
}

#[cfg(target_os = "linux")]
fn linux_set_current_affinity(cpu: usize) -> bool {
    let mut affinity = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    unsafe { libc::CPU_ZERO(&mut affinity) };
    unsafe { libc::CPU_SET(cpu, &mut affinity) };
    unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &affinity) == 0 }
}

#[cfg(target_os = "macos")]
fn macos_performance_cpu_count() -> Option<usize> {
    let mut value = 0i32;
    let mut size = std::mem::size_of_val(&value);
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.logicalcpu".as_ptr(),
            (&mut value as *mut i32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && value > 0).then_some(value as usize)
}

#[cfg(target_os = "macos")]
fn macos_select_performance_qos() -> bool {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_cpu_lists() {
        assert_eq!(
            parse_linux_cpu_list("0-3,8,10-11\n").expect("parse CPU list"),
            vec![0, 1, 2, 3, 8, 10, 11]
        );
        assert!(parse_linux_cpu_list("3-1").is_err());
        assert!(parse_linux_cpu_list("").is_err());
    }

    fn linux_cpu(
        cpu: usize,
        core_type: Option<u64>,
        package: Option<i64>,
        core: Option<i64>,
    ) -> LinuxCpuInfo {
        LinuxCpuInfo {
            cpu,
            core_type,
            capacity: None,
            max_frequency: None,
            physical_package_id: package,
            core_id: core,
        }
    }

    #[test]
    fn selects_one_logical_cpu_per_physical_performance_core() {
        let cpus = [
            linux_cpu(1, Some(2), Some(0), Some(0)),
            linux_cpu(0, Some(2), Some(0), Some(0)),
            linux_cpu(3, Some(2), Some(0), Some(1)),
            linux_cpu(2, Some(2), Some(0), Some(1)),
            linux_cpu(4, Some(1), Some(0), Some(2)),
            linux_cpu(6, Some(2), Some(1), Some(0)),
        ];

        assert_eq!(
            select_linux_performance_cpus(&cpus, &[0, 1, 2, 3, 4, 6])
                .expect("select performance CPUs"),
            vec![0, 2, 6]
        );
    }

    #[test]
    fn selection_respects_restricted_affinity_without_using_efficiency_cores() {
        let cpus = [
            linux_cpu(0, Some(2), Some(0), Some(0)),
            linux_cpu(1, Some(2), Some(0), Some(0)),
            linux_cpu(2, Some(2), Some(0), Some(1)),
            linux_cpu(4, Some(1), Some(0), Some(2)),
        ];

        assert_eq!(
            select_linux_performance_cpus(&cpus, &[1, 2, 4])
                .expect("select allowed performance CPUs"),
            vec![1, 2]
        );
        assert!(select_linux_performance_cpus(&cpus, &[4]).is_err());
    }

    #[test]
    fn selection_falls_back_when_classification_and_topology_are_absent() {
        let cpus = [
            linux_cpu(7, None, None, None),
            linux_cpu(2, None, None, None),
            linux_cpu(4, None, None, None),
        ];

        assert_eq!(
            select_linux_performance_cpus(&cpus, &[7, 2]).expect("select allowed CPUs"),
            vec![2, 7]
        );
    }

    #[test]
    fn selection_uses_capacity_when_core_type_is_absent() {
        let mut cpus = [
            linux_cpu(0, None, Some(0), Some(0)),
            linux_cpu(1, None, Some(0), Some(0)),
            linux_cpu(2, None, Some(0), Some(1)),
        ];
        cpus[0].capacity = Some(1024);
        cpus[1].capacity = Some(1024);
        cpus[2].capacity = Some(512);

        assert_eq!(
            select_linux_performance_cpus(&cpus, &[0, 1, 2]).expect("select capacity-ranked CPUs"),
            vec![0]
        );
    }

    #[test]
    fn homogeneous_complete_core_type_ignores_frequency_variation() {
        let mut cpus = [
            linux_cpu(0, Some(2), Some(0), Some(0)),
            linux_cpu(1, Some(2), Some(0), Some(1)),
        ];
        cpus[0].max_frequency = Some(4_800_000);
        cpus[1].max_frequency = Some(4_700_000);

        assert_eq!(
            select_linux_performance_cpus(&cpus, &[0, 1]).expect("select homogeneous CPUs"),
            vec![0, 1]
        );
    }

    #[test]
    fn maps_each_worker_to_one_distinct_cpu() {
        let cpus = [3, 7, 11];

        assert_eq!(linux_cpu_for_worker(&cpus, 0), Some(3));
        assert_eq!(linux_cpu_for_worker(&cpus, 1), Some(7));
        assert_eq!(linux_cpu_for_worker(&cpus, 2), Some(11));
        assert_eq!(linux_cpu_for_worker(&cpus, 3), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pool_workers_are_restricted_to_performance_cpus() {
        let expected = linux_performance_cpus().expect("detect performance CPUs");
        let pool = PerformanceCorePool::new("performance-test").expect("create performance pool");
        assert_eq!(pool.thread_count(), expected.len());

        let expected = expected
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let affinities = pool.inner.broadcast(|_| {
            let mut affinity = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
            assert_eq!(
                unsafe {
                    libc::sched_getaffinity(
                        0,
                        std::mem::size_of::<libc::cpu_set_t>(),
                        &mut affinity,
                    )
                },
                0
            );
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &affinity) })
                .collect::<std::collections::HashSet<_>>()
        });
        assert!(affinities.iter().all(|affinity| affinity.len() == 1));
        let assigned = affinities
            .into_iter()
            .flat_map(|affinity| affinity.into_iter())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(assigned, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn pool_uses_only_the_performance_cpu_count() {
        let expected = macos_performance_cpu_count().expect("detect performance CPU count");
        let pool = PerformanceCorePool::new("performance-test").expect("create performance pool");
        assert_eq!(pool.thread_count(), expected);
    }

    #[cfg(windows)]
    #[test]
    fn pool_uses_only_windows_performance_cpu_sets() {
        let expected = windows_performance_cpu_sets().expect("detect Windows performance CPUs");
        let pool = PerformanceCorePool::new("performance-test").expect("create performance pool");
        assert_eq!(pool.thread_count(), expected.len());
        assert!(pool
            .inner
            .broadcast(|_| windows_current_cpu_sets())
            .iter()
            .all(|selected| selected.as_deref() == Some(expected.as_slice())));
    }
}

#[cfg(all(test, windows))]
fn windows_current_cpu_sets() -> Option<Vec<u32>> {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadSelectedCpuSets};

    let mut required = 0u32;
    unsafe {
        GetThreadSelectedCpuSets(GetCurrentThread(), std::ptr::null_mut(), 0, &mut required);
    }
    if required == 0 {
        return None;
    }
    let capacity = usize::try_from(required).ok()?;
    let mut selected = vec![0u32; capacity];
    if unsafe {
        GetThreadSelectedCpuSets(
            GetCurrentThread(),
            selected.as_mut_ptr(),
            required,
            &mut required,
        )
    } == 0
    {
        return None;
    }
    selected.truncate(usize::try_from(required).ok()?);
    Some(selected)
}
