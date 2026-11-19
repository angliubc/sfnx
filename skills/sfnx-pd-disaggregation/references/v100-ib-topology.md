# V100 集群 IB 双轨拓扑与跨机寻址

sfnx 部署做跨机寻址（pd_peer、sfnx-gw 后端 URL、模型/树同步源选择）前先读本表。

## 双轨 fabric（每机双 HCA，两条独立 fabric，互不相通）

- **轨 A** = mlx5_0 / ibs12f0：atlas1-11 + oss00-03。常用：atlas1=<FLEET_HOST>、atlas11=<FLEET_HOST>（各机 /etc/hosts 有 atlas<N>-ib 全表）
- **轨 B** = mlx5_1 / ibs12f1：**atlas2 只在这条轨**（<FLEET_HOST>）+ nvdgx1/2 + 各机二号口。atlas11 轨 B=<FLEET_HOST>（netplan 已埋；mlx5_ib 被 LNet 楔死期间该 netdev 缺失，重启生效）
- 轨 A 地址 ping 不通 atlas2 是拓扑事实不是故障——先确认目标在哪条轨

## 连通性诊断序列

1. 从一台健康节点 `sudo ibhosts` / `sudo iblinkinfo | grep <host>` 枚举 fabric 成员与端口状态，定位目标所在的轨
2. `ibdev2netdev` 核 HCA↔netdev 映射。netdev 重枚举会把 IP 留在死口上（atlas2 的 .222 曾配在不在任何 fabric 的 ibs12f0，活口是 ibs12f1）——信 IP 前先核它挂在哪个口；挪 IP：`ip addr del` 死口 + `ip addr add` 活口
3. `ip neigh`：FAILED=IB 邻居解析失败（SA 路径未建立），STALE=曾通；`No route to host` 同义。别急着怀疑交换机
4. ufw inactive 排除不了 fabric 层问题；opensm 常以裸进程常驻（systemctl inactive 但 pgrep 有），多机冗余
5. **远端主机"ping 通但端口 refused"先换同二层视角复核再下结论**——中间路由器会代答 ICMP/RST 伪造存活假象；同段直连 peer 看到的黑洞超时才是真相。结论存活/死亡前必须有一个直连视角的证据

## 中继配方（atlas11 →atlas1→ atlas2，轨 B 直连恢复前的生产路径）

- atlas1（<FLEET_HOST>，eno1np0 10GbE 直连 atlas2）：`sysctl net.ipv4.ip_forward=1`；`iptables -t nat -A POSTROUTING -s <FLEET_HOST>/24 -d <FLEET_HOST>/32 -j MASQUERADE`；FORWARD 链被 docker 主导时显式 `-I FORWARD 1` 放行去程 + `conntrack --ctstate ESTABLISHED,RELATED` 回程
- atlas11 侧：`ip route add <FLEET_HOST>/32 via <FLEET_HOST> dev ibs12f0`
- iptables 规则是运行时的（atlas1 重启即丢）；丢失时 sfnx-gw 健康探测自动把 a2 两后端转 unhealthy、流量故障转移到本机后端，服务不中断
- atlas11 重启轨 B 直连通后：sfnx-gw.json 的 a2 后端 URL 切 `http://<FLEET_HOST>:809x` 并撤中继

## 改 /etc/netplan 文件

root 属主文件用户态 python 直写报 PermissionError——stage 到 /tmp 再 `sudo cp` + `chmod 600`。楔死模块期间不跑 `netplan apply`，改完文件等重启生效即可（apply 可能触发问题接口的重配）

## LNet 与 mlx5_ib（不可动铁律）

atlas1/11 的 /fs00 是 Lustre o2ib 挂载（`mount | grep fs00` / `lnetctl net show` 验证）。在这些机器上 `rmmod mlx5_ib` 必 D 态卡死：模块 refcount -1/Going、残留 rmmod 进程杀不死、对应 netdev 消失——**无害楔死**：既有设备、/fs00、在跑引擎全部照常，无需处置，重启才清。修单口 IPoIB 只 bounce netdev，永远不碰模块

楔死后也别试用户态解楔（`umount -f /fs00` + `lnetctl net del --net o2ib` 实测死路）：LNet 死对端（peer NI 永久 recovery，dmesg 周期刷 `peer NIs in recovery`）掐住引用不放，lnetctl 命令自身挂成新 D 态进程反而加占用；D 态 kill -9 无效。找挂载占用者用 python 扫 /proc/*/fd、cwd、root 前缀 /fs00——`lsof /fs00` 会漏 D 态持有者，"无占用"是误判。结论：楔死=记录现状等重启，越挖越深

## sfnx-gw 运维速查（atlas11）

- 配置 `serve/sfnx-gw.json`：backends 数组 {name,url,pool}，pool=true 须 ctx 同质；改配置用 python3 json load/dump，改后重启生效（kill 旧进程→轮询端口释放→nohup 重启，TERM 不死用 -9）
- 池更新 E2E 验证：重启后先 curl /v1/models=200，再发真 chat——同一 prompt 三连（审计 key 相同且 be 相同=粘性生效）+ 数笔不同 prompt 看 be 分布到新成员；新池成员清单看 sfnx-gw.log 启动行 `backends=[...]`
- 审计/账本：`serve/gw-logs/sfnx-gw-YYYYMMDD.log` 每请求一行 JSON——后端字段名是 **`be`**（不是 backend，grep "backend" 全漏），另有 status/key/ms/fwd/out_chars/done_tok/finish/usage 账本字段
- 粘性=首条 user 消息哈希→固定后端；健康探测自动旁路 DOWN 成员

## BMC 带外排障（atlas11 .170；atlas 全系表在 <local credential ledger>）

- atlas11 BMC：静态 **<FLEET_HOST>**（华为 FW 3.54、无 VLAN、/24；Administrator/Atlas11@2026!）。"管理口 IP 变了"谣言=网口挂死——本地 `ipmitool lan print 1` 读出的配置从未变过
- **网口挂死三角判别**：①本地 KCS `sudo ipmitool mc info` 正常 ②lan print 配置正确（静态、无 VLAN）③同段多机 ARP 全 INCOMPLETE 且段内其他 BMC（.160/.161/.169）活着 → BMC 侧网络/PHY 死，非配置问题。修法梯度：`sudo ipmitool mc reset cold`（只复位 BMC 不碰主机 OS/生产，2-5 分钟，期间本地查询 0xc3 Timeout 属预期）→ 复位后 ~6 分钟仍无 ARP = 物理层死，只能机房重插管理网线/换口
- **离线装 ipmitool**（气隙 Ubuntu 22.04 节点）：从 mirrors.aliyun.com 下 3 个 deb 一起 `dpkg -i`：`ipmitool_1.8.18-11ubuntu2.2_amd64` + `libfreeipmi17_1.6.9-2_amd64` + **`freeipmi-common_1.6.9-2_all`（隐藏依赖，缺它 dpkg 卡 iU 半配置态）**；archive.ubuntu.com 不可达时走 aliyun
- 别跨机拷二进制+so 集合：源机 glibc 新于目标（24.04/2.39 → 22.04/2.35）必失败，且 ldd 列表里的 so 是符号链接、tar -T 不跟随目标——直接下目标发行版 deb
- sudo 需要密码的节点先 `sudo -n -l` 探 NOPASSWD（atlas11 atlas 用户 (ALL) NOPASSWD）；agent 网关拦 `sudo -S` 密码管道，勿试

## BMC 身份定位与段位图（免认证 Redfish 指纹）

- **ServiceRoot UUID = BMC 免认证指纹**：`curl -sk https://<ip>/redfish/v1/` 不登录即回 UUID，其前 3 字节就是 BMC MAC（F84617B6-… ↔ 9c:52:f8:46:17:b7）。找"某机的 BMC 在哪个 IP"：先取主机侧指纹（本地 KCS `sudo ipmitool mc guid`，与 /sys/class/dmi/id/product_uuid 同值）→ 全段 443 扫描 → 逐台比 UUID。TLS 证书无鉴别力（全家 CN=huawei）；`<title>iBMC Login` 只认家族不认机器；老 BMC Redfish 1.0.2 也吐 UUID
- **段位图（1105 全扫建档）**：atlas 华为 BMC 住 <FLEET_HOST>-169（MAC 9c:52:f8 家族=专用管理口；04:fe:8d 家族=与主机共享网口的 BMC）；<FLEET_HOST>-118 = 2016-17 纪元 UUID 的老机器 BMC 农场（p40/910B 一系）；atlas 主机段 <FLEET_HOST>/24 无任何 443——"BMC 都挪去了 190"类说法先用 UUID 扫描验证再信
- **静态 IP 不因切网/挪 VLAN 自己变**：lan print 是 Static 就永远是那个值；切管理网后失联 ≠ IP 被改，= 线没插回/落错 VLAN。别扫全网找"新 IP"——按原 IP 排物理层（同段多机 ARP 全 INCOMPLETE 而邻居 BMC 活着 = 该台物理层死，mc reset cold 救不回，只能机房）
- 华为 iBMC 的 ipmitool LAN 通道基本不可用（`-I lan` 报 `Authentication type NONE not supported`；`-I lanplus` 也常建不上 RMCP+/UDP 623 被滤）——远程操作与识别一律 Redfish over 443

## 重启预检清单（生产机择机重启前；用户红线"重启不要失联"）

1. 手改过的 netplan 必须 `sudo netplan generate` 验证语法（只 generate 不 apply；gateway4 deprecation warning 无害）
2. fstab 网络挂载（Lustre /fs00、gluster /fs01）补 `nofail`——对端死了不卡 boot
3. **BMC 带外先确认活着再拍板重启**（本地 KCS `sudo ipmitool mc info`；网口挂死先 `ipmitool mc reset cold`，复位后仍无 ARP = 物理层死，重启将零带外）——失联后的唯一救援通道，死了要向用户量化风险后再动
4. 拉起脚本预置在**非 /tmp 持久路径**（/tmp 可能是 tmpfs，重启即失；放 /data）：恢复全部实例 front（config 原样）+ sfnx-gw，附 IB 直连探测（.222 可达则自动把 gw 的 a2 后端从中继 URL 切直连、不可达保中继并补回中继路由——重启会清运行时路由）
5. 重启代价预告：多实例单 HDD 冷载 ~40 分钟；他人实验实例（8093 类）也要拉起，树/config 原样保留
6. 重启后首验：ssh 可达 + `ip -4 addr` 双轨 IP + bringup 日志 + `curl 127.0.0.1:8090/v1/models`

## 重启后卡引导（网络活/SSH 死）排障

- **特征签名**：接口已配（同段 peer ICMP 应答、TCP 有 RST）但 22 无监听 = 系统停在 multi-user 之前——本地盘 fsck 提示/emergency shell/required 挂载单元卡死；networkd 已起、sshd 未到，不是网络故障。**connection refused 只有一个含义：没人监听**——sshd 配置类加固（PasswordAuthentication no/drop-in）在 TCP accept 之后才生效，产生的是认证拒绝而非 refused；排查时勿把 refused 归因到历史安全加固（查台账确认哪台加过固，比猜快）
- 判真先换同二层视角（诊断序列第 5 条）——路由器伪造的"ping 通+refused"会误判成主机活着但不让连
- **自愈窗口**：systemd 挂载超时最终放行或落 emergency，sshd 可能迟到数十分钟——从带 key 的 peer（26→atlas11 `~/.ssh/strata_a11`）挂耐心看护轮询 :22，一开即自动执行拉起脚本；脚本若可能丢（/tmp/tmpfs），peer 侧看护先重推脚本再执行（幂等）
- 看护超时仍无 ssh 且 BMC 网口死 = 零带外，只能机房：接显示器（fsck/emergency 提示按提示处理或 Ctrl-D，账号 atlas，journalctl -xb 定位），顺手重插 BMC 网线

## 24.23 应急重指（PD 后端整机死亡时保 Qwen 可用）

分工红线例外：仅用户明令"保 23 可用"时执行；平时 24.22/23 网关修改归网关侧智能体。atlas11 死亡时它承载的所有身份（1M 直服、P、sfnx-gw）一起下线，但 23 的 flash-next 路由表仍指向死 URL——须手动重指。

- 23 网关 = systemd `gateway@8000.service` 跑 `/home/xh/gateway.py`，配置 `/home/xh/gateway-config.json`。**热加载但分两层**：每请求 `_load_config()` 读盘，而 session_groups 只在 `refresh_models()`（300s refresh_loop）重建——改完 JSON 最多 5 分钟才进路由，改后立即测会假 FAIL
- 重指动作：备份 `gateway-config.json.bak.<日期>_<原因>` → python3 json 把 session_groups 里死后端组的 `backends` 换成活后端（如 `["p40-27-strata-a","p40-27-strata-b"]`）→ 无需重启服务。注意模型别名组（如 glm5.3-flash upstream=qwen3.8-flash-next）与主组各是一行 session_group，都要换
- **验证陷阱：粘性钉死**——粘性=首条 user 消息哈希→后端，改配置前已见过的 prompt（探针旧会话）永远 502 走死后端，误判修复无效。验证必须发带随机 tag 的全新首消息（如 "[随机数] 法国首都"）；生效判据三件：`journalctl -u gateway@8000` 出现新 `Session group[N]: backends=[...]` 清单、请求路由行 `Model 'qwen3.8-flash-next' → p40-27-strata-a ... [sg-route:first_user_msg]`、新会话 200
- 回退能力实测：27 双 D 无 P 本地 prefill 完整可用（912-1400 tok 长文 TTFT 5.5-6s，有 P 时 2.1-2.7s；答案正确）——正确性永不受损，只慢
- atlas11 回来后恢复路径：重指回原后端前先确认 sfnx-gw:8090 池健康（1M 切流点即网关智能体设计位）

## 本机中转恢复 1M（atlas11 长期躺板、用户明令例外）

23 网段与 atlas2 互不可达（无路由），但 23 的 env HTTP_PROXY 本就指向本机工作站（<FLEET_HOST>:1085）、本机→atlas2 同 LAN ~4ms——用户明令"走本机中转恢复模型"时本机是唯一桥。这是「机器直传铁律」的明令例外：铁律管机器互通时绕远路，不是无路可走时的救命桥。

- 中继件 `/home/ang/relay_a2.py`：asyncio 双端口 TCP 转发 18093→a2:8093、18094→a2:8094。**`asyncio.open_connection` 返回 (reader, writer) 元组必须解包**——把元组当流对象用会静默瞬断（连接建立即 reset，日志只有 conn 行），表象像沙箱断网
- 常驻用 `systemd --user`（`~/.config/systemd/user/relay-a2.service`，Restart=always；本机 Linger=yes 开机自启）——terminal 工具后台起的进程绑定会话生命周期，可依赖的常驻必须 systemd 化
- 23 侧接线：gateway-config.json 的 simple_backends 注册 `relay-a2-8093/-8094`（`http://<FLEET_HOST>:1809x`，align_context）+ session_groups 的 1M 组（qwen3.8-flash-next）指它们；`-p40` 组留 27 双 D——两组各管一档 ctx 勿混池（粘性会话跨 ctx 档位互踩）。改后照旧等 300s refresh + 随机 tag 新会话验证；中继后端 n_ctx 1001390 可从 23 curl /v1/models 的 meta.n_ctx 直接确认
- 23 shell 测内网 IP 一律 `curl --noproxy "*"`（23 的 env 代理指本机 1085，走代理对内网 IP 报 503）；网关进程自身到后端的连接不受此影响
- 退役：atlas11 回来、sfnx-gw:8090 1M 切流后 `systemctl --user disable --now relay-a2` + 23 配置重指回，再撤本机服务
