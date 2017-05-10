# 1 "bdaqctrl.h"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "bdaqctrl.h"
# 19 "bdaqctrl.h"

# 30 "bdaqctrl.h"
typedef signed char int8;
typedef signed short int16;

typedef unsigned char uint8;
typedef unsigned short uint16;
# 52 "bdaqctrl.h"
   typedef signed int int32;
   typedef unsigned int uint32;
   typedef signed long long int64;
   typedef unsigned long long uint64;
   typedef void * HANDLE;


typedef enum tagTerminalBoard {
   WiringBoard = 0,
   PCLD8710,
   PCLD789,
   PCLD8115,
} TerminalBoard;

typedef enum tagModuleType {
   DaqAny = -1,
   DaqGroup = 1,
   DaqDevice,
   DaqAi,
   DaqAo,
   DaqDio,
   DaqCounter,
} ModuleType;

typedef enum tagAccessMode {
   ModeRead = 0,
   ModeWrite,
   ModeWriteWithReset,
   ModeReserved = 0xffffffff,
} AccessMode;

typedef enum tagMathIntervalType {

   RightOpenSet = 0x0,
   RightClosedBoundary = 0x1,
   RightOpenBoundary = 0x2,


   LeftOpenSet = 0x0,
   LeftClosedBoundary = 0x4,
   LeftOpenBoundary = 0x8,


   Boundless = 0x0,


   LOSROS = 0x0,
   LOSRCB = 0x1,
   LOSROB = 0x2,

   LCBROS = 0x4,
   LCBRCB = 0x5,
   LCBROB = 0x6,

   LOBROS = 0x8,
   LOBRCB = 0x9,
   LOBROB = 0xA,
} MathIntervalType;

typedef struct tagMathInterval {
   int32 Type;
   double Min;
   double Max;
} MathInterval, * PMathInterval;

typedef enum tagAiChannelType {
   AllSingleEnded = 0,
   AllDifferential,
   AllSeDiffAdj,
   MixedSeDiffAdj,
} AiChannelType;

typedef enum tagAiSignalType {
   SingleEnded = 0,
   Differential,
} AiSignalType;

typedef enum tagFilterType {
   FilterNone = 0,
   LowPass,
   HighPass,
   BandPass,
   BandStop,
} FilterType;

typedef enum tagAiCouplingType {
   ACCoupling = 0,
   DCCoupling,
} AiCouplingType;

typedef enum tagAiImpedanceType {
   Ipd1Momh = 0,
   Ipd50omh,
} AiImpedanceType;

typedef enum tagDioPortType {
   PortDi = 0,
   PortDo,
   PortDio,
   Port8255A,
   Port8255C,
   PortIndvdlDio,
} DioPortType;

typedef enum tagDioPortDir {
   Input = 0x00,
   LoutHin = 0x0F,
   LinHout = 0xF0,
   Output = 0xFF,
} DioPortDir;

typedef enum tagDiOpenState {
   pullHighAllPort = 0x00,
   pullLowAllPort = 0x11,
} DiOpenState;

typedef enum tagSamplingMethod {
   EqualTimeSwitch = 0,
   Simultaneous,
} SamplingMethod;

typedef enum tagTemperatureDegree {
   Celsius = 0,
   Fahrenheit,
   Rankine,
   Kelvin,
} TemperatureDegree;

typedef enum tagBurnoutRetType {
   Current = 0,
   ParticularValue,
   UpLimit,
   LowLimit,
   LastCorrectValue,
} BurnoutRetType;

typedef enum tagValueUnit {
   Kilovolt,
   Volt,
   Millivolt,
   Microvolt,
   Kiloampere,
   Ampere,
   Milliampere,
   Microampere,
   CelsiusUnit,
} ValueUnit;

typedef enum tagValueRange {
   V_OMIT = -1,
   V_Neg15To15 = 0,
   V_Neg10To10,
   V_Neg5To5,
   V_Neg2pt5To2pt5,
   V_Neg1pt25To1pt25,
   V_Neg1To1,

   V_0To15,
   V_0To10,
   V_0To5,
   V_0To2pt5,
   V_0To1pt25,
   V_0To1,

   mV_Neg625To625,
   mV_Neg500To500,
   mV_Neg312pt5To312pt5,
   mV_Neg200To200,
   mV_Neg150To150,
   mV_Neg100To100,
   mV_Neg50To50,
   mV_Neg30To30,
   mV_Neg20To20,
   mV_Neg15To15,
   mV_Neg10To10,
   mV_Neg5To5,

   mV_0To625,
   mV_0To500,
   mV_0To150,
   mV_0To100,
   mV_0To50,
   mV_0To20,
   mV_0To15,
   mV_0To10,

   mA_Neg20To20,
   mA_0To20,
   mA_4To20,
   mA_0To24,


   V_Neg2To2,
   V_Neg4To4,
   V_Neg20To20,

   Jtype_0To760C = 0x8000,
   Ktype_0To1370C,
   Ttype_Neg100To400C,
   Etype_0To1000C,
   Rtype_500To1750C,
   Stype_500To1750C,
   Btype_500To1800C,

   Pt392_Neg50To150,
   Pt385_Neg200To200,
   Pt385_0To400,
   Pt385_Neg50To150,
   Pt385_Neg100To100,
   Pt385_0To100,
   Pt385_0To200,
   Pt385_0To600,
   Pt392_Neg100To100,
   Pt392_0To100,
   Pt392_0To200,
   Pt392_0To600,
   Pt392_0To400,
   Pt392_Neg200To200,
   Pt1000_Neg40To160,

   Balcon500_Neg30To120,

   Ni518_Neg80To100,
   Ni518_0To100,
   Ni508_0To100,
   Ni508_Neg50To200,

   Thermistor_3K_0To100,
   Thermistor_10K_0To100,

   Jtype_Neg210To1200C,
   Ktype_Neg270To1372C,
   Ttype_Neg270To400C,
   Etype_Neg270To1000C,
   Rtype_Neg50To1768C,
   Stype_Neg50To1768C,
   Btype_40To1820C,

   Jtype_Neg210To870C,
   Rtype_0To1768C,
   Stype_0To1768C,
   Ttype_Neg20To135C,


   UserCustomizedVrgStart = 0xC000,
   UserCustomizedVrgEnd = 0xF000,


   V_ExternalRefBipolar = 0xF001,
   V_ExternalRefUnipolar = 0xF002,
} ValueRange;

typedef enum tagSignalPolarity {
   Negative = 0,
   Positive,
} SignalPolarity;

typedef enum tagSignalCountingType {
   CountingNone = 0,
   DownCount,
   UpCount,
   PulseDirection,
   TwoPulse,
   AbPhaseX1,
   AbPhaseX2,
   AbPhaseX4,
} SignalCountingType;

typedef enum tagOutSignalType{
   SignalOutNone = 0,
   ChipDefined,
   NegChipDefined,
   PositivePulse,
   NegativePulse,
   ToggledFromLow,
   ToggledFromHigh,
} OutSignalType;

typedef enum tagCounterCapability {
   Primary = 1,
   InstantEventCount,
   OneShot,
   TimerPulse,
   InstantFreqMeter,
   InstantPwmIn,
   InstantPwmOut,
   UpDownCount,
} CounterCapability;

typedef enum tagCounterOperationMode {
   C8254_M0 = 0,
   C8254_M1,
   C8254_M2,
   C8254_M3,
   C8254_M4,
   C8254_M5,

   C1780_MA,
   C1780_MB,
   C1780_MC,
   C1780_MD,
   C1780_ME,
   C1780_MF,
   C1780_MG,
   C1780_MH,
   C1780_MI,
   C1780_MJ,
   C1780_MK,
   C1780_ML,
   C1780_MO,
   C1780_MR,
   C1780_MU,
   C1780_MX,
} CounterOperationMode;

typedef enum tagCounterValueRegister {
   CntLoad,
   CntPreset = CntLoad,
   CntHold,
   CntOverCompare,
   CntUnderCompare,
} CounterValueRegister;

typedef enum tagCounterCascadeGroup {
   GroupNone = 0,
   Cnt0Cnt1,
   Cnt2Cnt3,
   Cnt4Cnt5,
   Cnt6Cnt7,
} CounterCascadeGroup;

typedef enum tagFreqMeasureMethod {
   AutoAdaptive = 0,
   CountingPulseBySysTime,
   CountingPulseByDevTime,
   PeriodInverse,
} FreqMeasureMethod;

typedef enum tagActiveSignal {
   ActiveNone = 0,
   RisingEdge,
   FallingEdge,
   BothEdge,
   HighLevel,
   LowLevel,
} ActiveSignal;

typedef enum tagTriggerAction {
   ActionNone = 0,
   DelayToStart,
   DelayToStop,
} TriggerAction;

typedef enum tagSignalPosition {
   InternalSig = 0,
   OnConnector,
   OnAmsi,
} SignalPosition;

typedef enum tagSignalDrop {
   SignalNone = 0,


   SigInternalClock,
   SigInternal1KHz,
   SigInternal10KHz,
   SigInternal100KHz,
   SigInternal1MHz,
   SigInternal10MHz,
   SigInternal20MHz,
   SigInternal30MHz,
   SigInternal40MHz,
   SigInternal50MHz,
   SigInternal60MHz,

   SigDiPatternMatch,
   SigDiStatusChange,


   SigExtAnaClock,
   SigExtAnaScanClock,
   SigExtAnaTrigger,
   SigExtAnaTrigger0 = SigExtAnaTrigger,
   SigExtDigClock,
   SigExtDigTrigger0,
   SigExtDigTrigger1,
   SigExtDigTrigger2,
   SigExtDigTrigger3,
   SigCHFrzDo,



   SigAi0, SigAi1, SigAi2, SigAi3, SigAi4, SigAi5, SigAi6, SigAi7,
   SigAi8, SigAi9, SigAi10, SigAi11, SigAi12, SigAi13, SigAi14, SigAi15,
   SigAi16, SigAi17, SigAi18, SigAi19, SigAi20, SigAi21, SigAi22, SigAi23,
   SigAi24, SigAi25, SigAi26, SigAi27, SigAi28, SigAi29, SigAi30, SigAi31,
   SigAi32, SigAi33, SigAi34, SigAi35, SigAi36, SigAi37, SigAi38, SigAi39,
   SigAi40, SigAi41, SigAi42, SigAi43, SigAi44, SigAi45, SigAi46, SigAi47,
   SigAi48, SigAi49, SigAi50, SigAi51, SigAi52, SigAi53, SigAi54, SigAi55,
   SigAi56, SigAi57, SigAi58, SigAi59, SigAi60, SigAi61, SigAi62, SigAi63,


   SigAo0, SigAo1, SigAo2, SigAo3, SigAo4, SigAo5, SigAo6, SigAo7,
   SigAo8, SigAo9, SigAo10, SigAo11, SigAo12, SigAo13, SigAo14, SigAo15,
   SigAo16, SigAo17, SigAo18, SigAo19, SigAo20, SigAo21, SigAo22, SigAo23,
   SigAo24, SigAo25, SigAo26, SigAo27, SigAo28, SigAo29, SigAo30, SigAo31,


   SigDi0, SigDi1, SigDi2, SigDi3, SigDi4, SigDi5, SigDi6, SigDi7,
   SigDi8, SigDi9, SigDi10, SigDi11, SigDi12, SigDi13, SigDi14, SigDi15,
   SigDi16, SigDi17, SigDi18, SigDi19, SigDi20, SigDi21, SigDi22, SigDi23,
   SigDi24, SigDi25, SigDi26, SigDi27, SigDi28, SigDi29, SigDi30, SigDi31,
   SigDi32, SigDi33, SigDi34, SigDi35, SigDi36, SigDi37, SigDi38, SigDi39,
   SigDi40, SigDi41, SigDi42, SigDi43, SigDi44, SigDi45, SigDi46, SigDi47,
   SigDi48, SigDi49, SigDi50, SigDi51, SigDi52, SigDi53, SigDi54, SigDi55,
   SigDi56, SigDi57, SigDi58, SigDi59, SigDi60, SigDi61, SigDi62, SigDi63,
   SigDi64, SigDi65, SigDi66, SigDi67, SigDi68, SigDi69, SigDi70, SigDi71,
   SigDi72, SigDi73, SigDi74, SigDi75, SigDi76, SigDi77, SigDi78, SigDi79,
   SigDi80, SigDi81, SigDi82, SigDi83, SigDi84, SigDi85, SigDi86, SigDi87,
   SigDi88, SigDi89, SigDi90, SigDi91, SigDi92, SigDi93, SigDi94, SigDi95,
   SigDi96, SigDi97, SigDi98, SigDi99, SigDi100, SigDi101, SigDi102, SigDi103,
   SigDi104, SigDi105, SigDi106, SigDi107, SigDi108, SigDi109, SigDi110, SigDi111,
   SigDi112, SigDi113, SigDi114, SigDi115, SigDi116, SigDi117, SigDi118, SigDi119,
   SigDi120, SigDi121, SigDi122, SigDi123, SigDi124, SigDi125, SigDi126, SigDi127,
   SigDi128, SigDi129, SigDi130, SigDi131, SigDi132, SigDi133, SigDi134, SigDi135,
   SigDi136, SigDi137, SigDi138, SigDi139, SigDi140, SigDi141, SigDi142, SigDi143,
   SigDi144, SigDi145, SigDi146, SigDi147, SigDi148, SigDi149, SigDi150, SigDi151,
   SigDi152, SigDi153, SigDi154, SigDi155, SigDi156, SigDi157, SigDi158, SigDi159,
   SigDi160, SigDi161, SigDi162, SigDi163, SigDi164, SigDi165, SigDi166, SigDi167,
   SigDi168, SigDi169, SigDi170, SigDi171, SigDi172, SigDi173, SigDi174, SigDi175,
   SigDi176, SigDi177, SigDi178, SigDi179, SigDi180, SigDi181, SigDi182, SigDi183,
   SigDi184, SigDi185, SigDi186, SigDi187, SigDi188, SigDi189, SigDi190, SigDi191,
   SigDi192, SigDi193, SigDi194, SigDi195, SigDi196, SigDi197, SigDi198, SigDi199,
   SigDi200, SigDi201, SigDi202, SigDi203, SigDi204, SigDi205, SigDi206, SigDi207,
   SigDi208, SigDi209, SigDi210, SigDi211, SigDi212, SigDi213, SigDi214, SigDi215,
   SigDi216, SigDi217, SigDi218, SigDi219, SigDi220, SigDi221, SigDi222, SigDi223,
   SigDi224, SigDi225, SigDi226, SigDi227, SigDi228, SigDi229, SigDi230, SigDi231,
   SigDi232, SigDi233, SigDi234, SigDi235, SigDi236, SigDi237, SigDi238, SigDi239,
   SigDi240, SigDi241, SigDi242, SigDi243, SigDi244, SigDi245, SigDi246, SigDi247,
   SigDi248, SigDi249, SigDi250, SigDi251, SigDi252, SigDi253, SigDi254, SigDi255,


   SigDio0, SigDio1, SigDio2, SigDio3, SigDio4, SigDio5, SigDio6, SigDio7,
   SigDio8, SigDio9, SigDio10, SigDio11, SigDio12, SigDio13, SigDio14, SigDio15,
   SigDio16, SigDio17, SigDio18, SigDio19, SigDio20, SigDio21, SigDio22, SigDio23,
   SigDio24, SigDio25, SigDio26, SigDio27, SigDio28, SigDio29, SigDio30, SigDio31,
   SigDio32, SigDio33, SigDio34, SigDio35, SigDio36, SigDio37, SigDio38, SigDio39,
   SigDio40, SigDio41, SigDio42, SigDio43, SigDio44, SigDio45, SigDio46, SigDio47,
   SigDio48, SigDio49, SigDio50, SigDio51, SigDio52, SigDio53, SigDio54, SigDio55,
   SigDio56, SigDio57, SigDio58, SigDio59, SigDio60, SigDio61, SigDio62, SigDio63,
   SigDio64, SigDio65, SigDio66, SigDio67, SigDio68, SigDio69, SigDio70, SigDio71,
   SigDio72, SigDio73, SigDio74, SigDio75, SigDio76, SigDio77, SigDio78, SigDio79,
   SigDio80, SigDio81, SigDio82, SigDio83, SigDio84, SigDio85, SigDio86, SigDio87,
   SigDio88, SigDio89, SigDio90, SigDio91, SigDio92, SigDio93, SigDio94, SigDio95,
   SigDio96, SigDio97, SigDio98, SigDio99, SigDio100, SigDio101, SigDio102, SigDio103,
   SigDio104, SigDio105, SigDio106, SigDio107, SigDio108, SigDio109, SigDio110, SigDio111,
   SigDio112, SigDio113, SigDio114, SigDio115, SigDio116, SigDio117, SigDio118, SigDio119,
   SigDio120, SigDio121, SigDio122, SigDio123, SigDio124, SigDio125, SigDio126, SigDio127,
   SigDio128, SigDio129, SigDio130, SigDio131, SigDio132, SigDio133, SigDio134, SigDio135,
   SigDio136, SigDio137, SigDio138, SigDio139, SigDio140, SigDio141, SigDio142, SigDio143,
   SigDio144, SigDio145, SigDio146, SigDio147, SigDio148, SigDio149, SigDio150, SigDio151,
   SigDio152, SigDio153, SigDio154, SigDio155, SigDio156, SigDio157, SigDio158, SigDio159,
   SigDio160, SigDio161, SigDio162, SigDio163, SigDio164, SigDio165, SigDio166, SigDio167,
   SigDio168, SigDio169, SigDio170, SigDio171, SigDio172, SigDio173, SigDio174, SigDio175,
   SigDio176, SigDio177, SigDio178, SigDio179, SigDio180, SigDio181, SigDio182, SigDio183,
   SigDio184, SigDio185, SigDio186, SigDio187, SigDio188, SigDio189, SigDio190, SigDio191,
   SigDio192, SigDio193, SigDio194, SigDio195, SigDio196, SigDio197, SigDio198, SigDio199,
   SigDio200, SigDio201, SigDio202, SigDio203, SigDio204, SigDio205, SigDio206, SigDio207,
   SigDio208, SigDio209, SigDio210, SigDio211, SigDio212, SigDio213, SigDio214, SigDio215,
   SigDio216, SigDio217, SigDio218, SigDio219, SigDio220, SigDio221, SigDio222, SigDio223,
   SigDio224, SigDio225, SigDio226, SigDio227, SigDio228, SigDio229, SigDio230, SigDio231,
   SigDio232, SigDio233, SigDio234, SigDio235, SigDio236, SigDio237, SigDio238, SigDio239,
   SigDio240, SigDio241, SigDio242, SigDio243, SigDio244, SigDio245, SigDio246, SigDio247,
   SigDio248, SigDio249, SigDio250, SigDio251, SigDio252, SigDio253, SigDio254, SigDio255,


   SigCntClk0, SigCntClk1, SigCntClk2, SigCntClk3, SigCntClk4, SigCntClk5, SigCntClk6, SigCntClk7,


   SigCntGate0, SigCntGate1, SigCntGate2, SigCntGate3, SigCntGate4, SigCntGate5, SigCntGate6, SigCntGate7,


   SigCntOut0, SigCntOut1, SigCntOut2, SigCntOut3, SigCntOut4, SigCntOut5, SigCntOut6, SigCntOut7,


   SigCntFout0, SigCntFout1, SigCntFout2, SigCntFout3, SigCntFout4, SigCntFout5, SigCntFout6, SigCntFout7,


   SigAmsiPin0, SigAmsiPin1, SigAmsiPin2, SigAmsiPin3, SigAmsiPin4, SigAmsiPin5, SigAmsiPin6, SigAmsiPin7,
   SigAmsiPin8, SigAmsiPin9, SigAmsiPin10, SigAmsiPin11, SigAmsiPin12, SigAmsiPin13, SigAmsiPin14, SigAmsiPin15,
   SigAmsiPin16, SigAmsiPin17, SigAmsiPin18, SigAmsiPin19,


   SigInternal2Hz,
   SigInternal20Hz,
   SigInternal200Hz,
   SigInternal2KHz,
   SigInternal20KHz,
   SigInternal200KHz,
   SigInternal2MHz,


   SigExtAnaTrigger1,


   SigExtDigRefClock,
} SignalDrop;




typedef enum tagEventId {
   EvtDeviceRemoved = 0,
   EvtDeviceReconnected,
   EvtPropertyChanged,



   EvtBufferedAiDataReady,
   EvtBufferedAiOverrun,
   EvtBufferedAiCacheOverflow,
   EvtBufferedAiStopped,




   EvtBufferedAoDataTransmitted,
   EvtBufferedAoUnderrun,
   EvtBufferedAoCacheEmptied,
   EvtBufferedAoTransStopped,
   EvtBufferedAoStopped,




   EvtDiintChannel000, EvtDiintChannel001, EvtDiintChannel002, EvtDiintChannel003,
   EvtDiintChannel004, EvtDiintChannel005, EvtDiintChannel006, EvtDiintChannel007,
   EvtDiintChannel008, EvtDiintChannel009, EvtDiintChannel010, EvtDiintChannel011,
   EvtDiintChannel012, EvtDiintChannel013, EvtDiintChannel014, EvtDiintChannel015,
   EvtDiintChannel016, EvtDiintChannel017, EvtDiintChannel018, EvtDiintChannel019,
   EvtDiintChannel020, EvtDiintChannel021, EvtDiintChannel022, EvtDiintChannel023,
   EvtDiintChannel024, EvtDiintChannel025, EvtDiintChannel026, EvtDiintChannel027,
   EvtDiintChannel028, EvtDiintChannel029, EvtDiintChannel030, EvtDiintChannel031,
   EvtDiintChannel032, EvtDiintChannel033, EvtDiintChannel034, EvtDiintChannel035,
   EvtDiintChannel036, EvtDiintChannel037, EvtDiintChannel038, EvtDiintChannel039,
   EvtDiintChannel040, EvtDiintChannel041, EvtDiintChannel042, EvtDiintChannel043,
   EvtDiintChannel044, EvtDiintChannel045, EvtDiintChannel046, EvtDiintChannel047,
   EvtDiintChannel048, EvtDiintChannel049, EvtDiintChannel050, EvtDiintChannel051,
   EvtDiintChannel052, EvtDiintChannel053, EvtDiintChannel054, EvtDiintChannel055,
   EvtDiintChannel056, EvtDiintChannel057, EvtDiintChannel058, EvtDiintChannel059,
   EvtDiintChannel060, EvtDiintChannel061, EvtDiintChannel062, EvtDiintChannel063,
   EvtDiintChannel064, EvtDiintChannel065, EvtDiintChannel066, EvtDiintChannel067,
   EvtDiintChannel068, EvtDiintChannel069, EvtDiintChannel070, EvtDiintChannel071,
   EvtDiintChannel072, EvtDiintChannel073, EvtDiintChannel074, EvtDiintChannel075,
   EvtDiintChannel076, EvtDiintChannel077, EvtDiintChannel078, EvtDiintChannel079,
   EvtDiintChannel080, EvtDiintChannel081, EvtDiintChannel082, EvtDiintChannel083,
   EvtDiintChannel084, EvtDiintChannel085, EvtDiintChannel086, EvtDiintChannel087,
   EvtDiintChannel088, EvtDiintChannel089, EvtDiintChannel090, EvtDiintChannel091,
   EvtDiintChannel092, EvtDiintChannel093, EvtDiintChannel094, EvtDiintChannel095,
   EvtDiintChannel096, EvtDiintChannel097, EvtDiintChannel098, EvtDiintChannel099,
   EvtDiintChannel100, EvtDiintChannel101, EvtDiintChannel102, EvtDiintChannel103,
   EvtDiintChannel104, EvtDiintChannel105, EvtDiintChannel106, EvtDiintChannel107,
   EvtDiintChannel108, EvtDiintChannel109, EvtDiintChannel110, EvtDiintChannel111,
   EvtDiintChannel112, EvtDiintChannel113, EvtDiintChannel114, EvtDiintChannel115,
   EvtDiintChannel116, EvtDiintChannel117, EvtDiintChannel118, EvtDiintChannel119,
   EvtDiintChannel120, EvtDiintChannel121, EvtDiintChannel122, EvtDiintChannel123,
   EvtDiintChannel124, EvtDiintChannel125, EvtDiintChannel126, EvtDiintChannel127,
   EvtDiintChannel128, EvtDiintChannel129, EvtDiintChannel130, EvtDiintChannel131,
   EvtDiintChannel132, EvtDiintChannel133, EvtDiintChannel134, EvtDiintChannel135,
   EvtDiintChannel136, EvtDiintChannel137, EvtDiintChannel138, EvtDiintChannel139,
   EvtDiintChannel140, EvtDiintChannel141, EvtDiintChannel142, EvtDiintChannel143,
   EvtDiintChannel144, EvtDiintChannel145, EvtDiintChannel146, EvtDiintChannel147,
   EvtDiintChannel148, EvtDiintChannel149, EvtDiintChannel150, EvtDiintChannel151,
   EvtDiintChannel152, EvtDiintChannel153, EvtDiintChannel154, EvtDiintChannel155,
   EvtDiintChannel156, EvtDiintChannel157, EvtDiintChannel158, EvtDiintChannel159,
   EvtDiintChannel160, EvtDiintChannel161, EvtDiintChannel162, EvtDiintChannel163,
   EvtDiintChannel164, EvtDiintChannel165, EvtDiintChannel166, EvtDiintChannel167,
   EvtDiintChannel168, EvtDiintChannel169, EvtDiintChannel170, EvtDiintChannel171,
   EvtDiintChannel172, EvtDiintChannel173, EvtDiintChannel174, EvtDiintChannel175,
   EvtDiintChannel176, EvtDiintChannel177, EvtDiintChannel178, EvtDiintChannel179,
   EvtDiintChannel180, EvtDiintChannel181, EvtDiintChannel182, EvtDiintChannel183,
   EvtDiintChannel184, EvtDiintChannel185, EvtDiintChannel186, EvtDiintChannel187,
   EvtDiintChannel188, EvtDiintChannel189, EvtDiintChannel190, EvtDiintChannel191,
   EvtDiintChannel192, EvtDiintChannel193, EvtDiintChannel194, EvtDiintChannel195,
   EvtDiintChannel196, EvtDiintChannel197, EvtDiintChannel198, EvtDiintChannel199,
   EvtDiintChannel200, EvtDiintChannel201, EvtDiintChannel202, EvtDiintChannel203,
   EvtDiintChannel204, EvtDiintChannel205, EvtDiintChannel206, EvtDiintChannel207,
   EvtDiintChannel208, EvtDiintChannel209, EvtDiintChannel210, EvtDiintChannel211,
   EvtDiintChannel212, EvtDiintChannel213, EvtDiintChannel214, EvtDiintChannel215,
   EvtDiintChannel216, EvtDiintChannel217, EvtDiintChannel218, EvtDiintChannel219,
   EvtDiintChannel220, EvtDiintChannel221, EvtDiintChannel222, EvtDiintChannel223,
   EvtDiintChannel224, EvtDiintChannel225, EvtDiintChannel226, EvtDiintChannel227,
   EvtDiintChannel228, EvtDiintChannel229, EvtDiintChannel230, EvtDiintChannel231,
   EvtDiintChannel232, EvtDiintChannel233, EvtDiintChannel234, EvtDiintChannel235,
   EvtDiintChannel236, EvtDiintChannel237, EvtDiintChannel238, EvtDiintChannel239,
   EvtDiintChannel240, EvtDiintChannel241, EvtDiintChannel242, EvtDiintChannel243,
   EvtDiintChannel244, EvtDiintChannel245, EvtDiintChannel246, EvtDiintChannel247,
   EvtDiintChannel248, EvtDiintChannel249, EvtDiintChannel250, EvtDiintChannel251,
   EvtDiintChannel252, EvtDiintChannel253, EvtDiintChannel254, EvtDiintChannel255,

   EvtDiCosintPort000, EvtDiCosintPort001, EvtDiCosintPort002, EvtDiCosintPort003,
   EvtDiCosintPort004, EvtDiCosintPort005, EvtDiCosintPort006, EvtDiCosintPort007,
   EvtDiCosintPort008, EvtDiCosintPort009, EvtDiCosintPort010, EvtDiCosintPort011,
   EvtDiCosintPort012, EvtDiCosintPort013, EvtDiCosintPort014, EvtDiCosintPort015,
   EvtDiCosintPort016, EvtDiCosintPort017, EvtDiCosintPort018, EvtDiCosintPort019,
   EvtDiCosintPort020, EvtDiCosintPort021, EvtDiCosintPort022, EvtDiCosintPort023,
   EvtDiCosintPort024, EvtDiCosintPort025, EvtDiCosintPort026, EvtDiCosintPort027,
   EvtDiCosintPort028, EvtDiCosintPort029, EvtDiCosintPort030, EvtDiCosintPort031,

   EvtDiPmintPort000, EvtDiPmintPort001, EvtDiPmintPort002, EvtDiPmintPort003,
   EvtDiPmintPort004, EvtDiPmintPort005, EvtDiPmintPort006, EvtDiPmintPort007,
   EvtDiPmintPort008, EvtDiPmintPort009, EvtDiPmintPort010, EvtDiPmintPort011,
   EvtDiPmintPort012, EvtDiPmintPort013, EvtDiPmintPort014, EvtDiPmintPort015,
   EvtDiPmintPort016, EvtDiPmintPort017, EvtDiPmintPort018, EvtDiPmintPort019,
   EvtDiPmintPort020, EvtDiPmintPort021, EvtDiPmintPort022, EvtDiPmintPort023,
   EvtDiPmintPort024, EvtDiPmintPort025, EvtDiPmintPort026, EvtDiPmintPort027,
   EvtDiPmintPort028, EvtDiPmintPort029, EvtDiPmintPort030, EvtDiPmintPort031,

   EvtBufferedDiDataReady,
   EvtBufferedDiOverrun,
   EvtBufferedDiCacheOverflow,
   EvtBufferedDiStopped,

   EvtBufferedDoDataTransmitted,
   EvtBufferedDoUnderrun,
   EvtBufferedDoCacheEmptied,
   EvtBufferedDoTransStopped,
   EvtBufferedDoStopped,

   EvtReflectWdtOccured,



   EvtCntTerminalCount0, EvtCntTerminalCount1, EvtCntTerminalCount2, EvtCntTerminalCount3,
   EvtCntTerminalCount4, EvtCntTerminalCount5, EvtCntTerminalCount6, EvtCntTerminalCount7,

   EvtCntOverCompare0, EvtCntOverCompare1, EvtCntOverCompare2, EvtCntOverCompare3,
   EvtCntOverCompare4, EvtCntOverCompare5, EvtCntOverCompare6, EvtCntOverCompare7,

   EvtCntUnderCompare0, EvtCntUnderCompare1, EvtCntUnderCompare2, EvtCntUnderCompare3,
   EvtCntUnderCompare4, EvtCntUnderCompare5, EvtCntUnderCompare6, EvtCntUnderCompare7,

   EvtCntEcOverCompare0, EvtCntEcOverCompare1, EvtCntEcOverCompare2, EvtCntEcOverCompare3,
   EvtCntEcOverCompare4, EvtCntEcOverCompare5, EvtCntEcOverCompare6, EvtCntEcOverCompare7,

   EvtCntEcUnderCompare0, EvtCntEcUnderCompare1, EvtCntEcUnderCompare2, EvtCntEcUnderCompare3,
   EvtCntEcUnderCompare4, EvtCntEcUnderCompare5, EvtCntEcUnderCompare6, EvtCntEcUnderCompare7,

   EvtCntOneShot0, EvtCntOneShot1, EvtCntOneShot2, EvtCntOneShot3,
   EvtCntOneShot4, EvtCntOneShot5, EvtCntOneShot6, EvtCntOneShot7,

   EvtCntTimer0, EvtCntTimer1, EvtCntTimer2, EvtCntTimer3,
   EvtCntTimer4, EvtCntTimer5, EvtCntTimer6, EvtCntTimer7,

   EvtCntPwmInOverflow0, EvtCntPwmInOverflow1, EvtCntPwmInOverflow2, EvtCntPwmInOverflow3,
   EvtCntPwmInOverflow4, EvtCntPwmInOverflow5, EvtCntPwmInOverflow6, EvtCntPwmInOverflow7,

   EvtUdIndex0, EvtUdIndex1, EvtUdIndex2, EvtUdIndex3,
   EvtUdIndex4, EvtUdIndex5, EvtUdIndex6, EvtUdIndex7,

   EvtCntPatternMatch0, EvtCntPatternMatch1, EvtCntPatternMatch2, EvtCntPatternMatch3,
   EvtCntPatternMatch4, EvtCntPatternMatch5, EvtCntPatternMatch6, EvtCntPatternMatch7,

   EvtCntCompareTableEnd0, EvtCntCompareTableEnd1, EvtCntCompareTableEnd2, EvtCntCompareTableEnd3,
   EvtCntCompareTableEnd4, EvtCntCompareTableEnd5, EvtCntCompareTableEnd6, EvtCntCompareTableEnd7,




   EvtBufferedAiRecordReady,
} EventId ;




typedef enum tagPropertyAttribute {
   ReadOnly = 0,
   Writable = 1,
   Modal = 0,
   Nature = 2,
} PropertyAttribute;

typedef enum tagPropertyId {



   CFG_Number,
   CFG_ComponentType,
   CFG_Description,
   CFG_Parent,
   CFG_ChildList,




   CFG_DevicesNumber,
   CFG_DevicesHandle,




   CFG_DeviceGroupNumber,
   CFG_DeviceProductID,
   CFG_DeviceBoardID,
   CFG_DeviceBoardVersion,
   CFG_DeviceDriverVersion,
   CFG_DeviceDllVersion,
   CFG_DeviceLocation,
   CFG_DeviceBaseAddresses,
   CFG_DeviceInterrupts,
   CFG_DeviceSupportedTerminalBoardTypes,
   CFG_DeviceTerminalBoardType,
   CFG_DeviceSupportedEvents,
   CFG_DeviceHotResetPreventable,
   CFG_DeviceLoadingTimeInit,
   CFG_DeviceWaitingForReconnect,
   CFG_DeviceWaitingForSleep,




   CFG_FeatureResolutionInBit,
   CFG_FeatureDataSize,
   CFG_FeatureDataMask,
   CFG_FeatureChannelNumberMax,
   CFG_FeatureChannelConnectionType,
   CFG_FeatureBurnDetectedReturnTypes,
   CFG_FeatureBurnoutDetectionChannels,
   CFG_FeatureOverallVrgType,
   CFG_FeatureVrgTypes,
   CFG_FeatureExtRefRange,
   CFG_FeatureExtRefAntiPolar,
   CFG_FeatureCjcChannels,
   CFG_FeatureChannelScanMethod,
   CFG_FeatureScanChannelStartBase,
   CFG_FeatureScanChannelCountBase,
   CFG_FeatureConvertClockSources,
   CFG_FeatureConvertClockRateRange,
   CFG_FeatureScanClockSources,
   CFG_FeatureScanClockRateRange,
   CFG_FeatureScanCountMax,
   CFG_FeatureTriggersCount,
   CFG_FeatureTriggerSources,
   CFG_FeatureTriggerActions,
   CFG_FeatureTriggerDelayCountRange,
   CFG_FeatureTriggerSources1,
   CFG_FeatureTriggerActions1,
   CFG_FeatureTriggerDelayCountRange1,

   CFG_ChannelCount,
   CFG_ConnectionTypeOfChannels,
   CFG_VrgTypeOfChannels,
   CFG_BurnDetectedReturnTypeOfChannels,
   CFG_BurnoutReturnValueOfChannels,
   CFG_ExtRefValueForUnipolar,
   CFG_ExtRefValueForBipolar,

   CFG_CjcChannel,
   CFG_CjcUpdateFrequency,
   CFG_CjcValue,

   CFG_SectionDataCount,
   CFG_ConvertClockSource,
   CFG_ConvertClockRatePerChannel,
   CFG_ScanChannelStart,
   CFG_ScanChannelCount,
   CFG_ScanClockSource,
   CFG_ScanClockRate,
   CFG_ScanCount,
   CFG_TriggerSource,
   CFG_TriggerSourceEdge,
   CFG_TriggerSourceLevel,
   CFG_TriggerDelayCount,
   CFG_TriggerAction,
   CFG_TriggerSource1,
   CFG_TriggerSourceEdge1,
   CFG_TriggerSourceLevel1,
   CFG_TriggerDelayCount1,
   CFG_TriggerAction1,
   CFG_ParentSignalConnectionChannel,
   CFG_ParentCjcConnectionChannel,
   CFG_ParentControlPort,




   CFG_FeaturePortsCount,
   CFG_FeaturePortsType,
   CFG_FeatureNoiseFilterOfChannels,
   CFG_FeatureNoiseFilterBlockTimeRange,
   CFG_FeatureDiintTriggerEdges,
   CFG_FeatureDiintOfChannels,
   CFG_FeatureDiintGateOfChannels,
   CFG_FeatureDiCosintOfChannels,
   CFG_FeatureDiPmintOfChannels,
   CFG_FeatureSnapEventSources,
   CFG_FeatureDiSnapEventSources = CFG_FeatureSnapEventSources,
   CFG_FeatureDoFreezeSignalSources,
   CFG_FeatureDoReflectWdtFeedIntervalRange,

   CFG_FeatureDiPortScanMethod,
   CFG_FeatureDiConvertClockSources,
   CFG_FeatureDiConvertClockRateRange,
   CFG_FeatureDiScanClockSources,
   CFG_FeatureDiScanClockRateRange,
   CFG_FeatureDiScanCountMax,
   CFG_FeatureDiTriggersCount,
   CFG_FeatureDiTriggerSources,
   CFG_FeatureDiTriggerActions,
   CFG_FeatureDiTriggerDelayCountRange,
   CFG_FeatureDiTriggerSources1,
   CFG_FeatureDiTriggerActions1,
   CFG_FeatureDiTriggerDelayCountRange1,

   CFG_FeatureDoPortScanMethod,
   CFG_FeatureDoConvertClockSources,
   CFG_FeatureDoConvertClockRateRange,
   CFG_FeatureDoScanClockSources,
   CFG_FeatureDoScanClockRateRange,
   CFG_FeatureDoScanCountMax,
   CFG_FeatureDoTriggersCount,
   CFG_FeatureDoTriggerSources,
   CFG_FeatureDoTriggerActions,
   CFG_FeatureDoTriggerDelayCountRange,
   CFG_FeatureDoTriggerSources1,
   CFG_FeatureDoTriggerActions1,
   CFG_FeatureDoTriggerDelayCountRange1,

   CFG_DirectionOfPorts,
   CFG_DiDataMaskOfPorts,
   CFG_DoDataMaskOfPorts,

   CFG_NoiseFilterOverallBlockTime,
   CFG_NoiseFilterEnabledChannels,
   CFG_DiintTriggerEdgeOfChannels,
   CFG_DiintGateEnabledChannels,
   CFG_DiCosintEnabledChannels,
   CFG_DiPmintEnabledChannels,
   CFG_DiPmintValueOfPorts,
   CFG_DoInitialStateOfPorts,
   CFG_DoFreezeEnabled,
   CFG_DoFreezeSignalState,
   CFG_DoReflectWdtFeedInterval,
   CFG_DoReflectWdtLockValue,
   CFG_DiSectionDataCount,
   CFG_DiConvertClockSource,
   CFG_DiConvertClockRatePerPort,
   CFG_DiScanPortStart,
   CFG_DiScanPortCount,
   CFG_DiScanClockSource,
   CFG_DiScanClockRate,
   CFG_DiScanCount,
   CFG_DiTriggerAction,
   CFG_DiTriggerSource,
   CFG_DiTriggerSourceEdge,
   CFG_DiTriggerSourceLevel,
   CFG_DiTriggerDelayCount,
   CFG_DiTriggerAction1,
   CFG_DiTriggerSource1,
   CFG_DiTriggerSourceEdge1,
   CFG_DiTriggerSourceLevel1,
   CFG_DiTriggerDelayCount1,

   CFG_DoSectionDataCount,
   CFG_DoConvertClockSource,
   CFG_DoConvertClockRatePerPort,
   CFG_DoScanPortStart,
   CFG_DoScanPortCount,
   CFG_DoScanClockSource,
   CFG_DoScanClockRate,
   CFG_DoScanCount,
   CFG_DoTriggerAction,
   CFG_DoTriggerSource,
   CFG_DoTriggerSourceEdge,
   CFG_DoTriggerSourceLevel,
   CFG_DoTriggerDelayCount,
   CFG_DoTriggerAction1,
   CFG_DoTriggerSource1,
   CFG_DoTriggerSourceEdge1,
   CFG_DoTriggerSourceLevel1,
   CFG_DoTriggerDelayCount1,





   CFG_FeatureCapabilitiesOfCounter0 = 174,
   CFG_FeatureCapabilitiesOfCounter1,
   CFG_FeatureCapabilitiesOfCounter2,
   CFG_FeatureCapabilitiesOfCounter3,
   CFG_FeatureCapabilitiesOfCounter4,
   CFG_FeatureCapabilitiesOfCounter5,
   CFG_FeatureCapabilitiesOfCounter6,
   CFG_FeatureCapabilitiesOfCounter7,


   CFG_FeatureChipOperationModes = 206,
   CFG_FeatureChipSignalCountingTypes,


   CFG_FeatureTmrCascadeGroups = 211,


   CFG_FeatureFmMethods = 213,


   CFG_ChipOperationModeOfCounters = 220,
   CFG_ChipSignalCountingTypeOfCounters,
   CFG_ChipLoadValueOfCounters,
   CFG_ChipHoldValueOfCounters,
   CFG_ChipOverCompareValueOfCounters,
   CFG_ChipUnderCompareValueOfCounters,
   CFG_ChipOverCompareEnabledCounters,
   CFG_ChipUnderCompareEnabledCounters,


   CFG_FmMethodOfCounters = 231,
   CFG_FmCollectionPeriodOfCounters,




   CFG_DevicePrivateRegionLength,
   CFG_SaiAutoConvertClockRate,
   CFG_SaiAutoConvertChannelStart,
   CFG_SaiAutoConvertChannelCount,
   CFG_ExtPauseSignalEnabled,
   CFG_ExtPauseSignalPolarity,
   CFG_OrderOfChannels,
   CFG_InitialStateOfChannels,





   CFG_FeatureChipClkSourceOfCounter0 = 242,
   CFG_FeatureChipClkSourceOfCounter1,
   CFG_FeatureChipClkSourceOfCounter2,
   CFG_FeatureChipClkSourceOfCounter3,
   CFG_FeatureChipClkSourceOfCounter4,
   CFG_FeatureChipClkSourceOfCounter5,
   CFG_FeatureChipClkSourceOfCounter6,
   CFG_FeatureChipClkSourceOfCounter7,

   CFG_FeatureChipGateSourceOfCounter0,
   CFG_FeatureChipGateSourceOfCounter1,
   CFG_FeatureChipGateSourceOfCounter2,
   CFG_FeatureChipGateSourceOfCounter3,
   CFG_FeatureChipGateSourceOfCounter4,
   CFG_FeatureChipGateSourceOfCounter5,
   CFG_FeatureChipGateSourceOfCounter6,
   CFG_FeatureChipGateSourceOfCounter7,

   CFG_FeatureChipValueRegisters,


   CFG_FeatureOsClkSourceOfCounter0,
   CFG_FeatureOsClkSourceOfCounter1,
   CFG_FeatureOsClkSourceOfCounter2,
   CFG_FeatureOsClkSourceOfCounter3,
   CFG_FeatureOsClkSourceOfCounter4,
   CFG_FeatureOsClkSourceOfCounter5,
   CFG_FeatureOsClkSourceOfCounter6,
   CFG_FeatureOsClkSourceOfCounter7,

   CFG_FeatureOsGateSourceOfCounter0,
   CFG_FeatureOsGateSourceOfCounter1,
   CFG_FeatureOsGateSourceOfCounter2,
   CFG_FeatureOsGateSourceOfCounter3,
   CFG_FeatureOsGateSourceOfCounter4,
   CFG_FeatureOsGateSourceOfCounter5,
   CFG_FeatureOsGateSourceOfCounter6,
   CFG_FeatureOsGateSourceOfCounter7,


   CFG_FeaturePiCascadeGroups,


   CFG_ChipClkSourceOfCounters = 279,
   CFG_ChipGateSourceOfCounters,


   CFG_OsClkSourceOfCounters,
   CFG_OsGateSourceOfCounters,
   CFG_OsDelayCountOfCounters,


   CFG_TmrFrequencyOfCounters,


   CFG_PoHiPeriodOfCounters,
   CFG_PoLoPeriodOfCounters,





   CFG_FeatureEcClkPolarities,
   CFG_FeatureEcGatePolarities,
   CFG_FeatureEcGateControlOfCounters,

   CFG_EcClkPolarityOfCounters,
   CFG_EcGatePolarityOfCounters,
   CFG_EcGateEnabledOfCounters,


   CFG_FeatureOsClkPolarities,
   CFG_FeatureOsGatePolarities,
   CFG_FeatureOsOutSignals,

   CFG_OsClkPolarityOfCounters,
   CFG_OsGatePolarityOfCounters,
   CFG_OsOutSignalOfCounters,


   CFG_FeatureTmrGateControlOfCounters,
   CFG_FeatureTmrGatePolarities,
   CFG_FeatureTmrOutSignals,
   CFG_FeatureTmrFrequencyRange,

   CFG_TmrGateEnabledOfCounters,
   CFG_TmrGatePolarityOfCounters,
   CFG_TmrOutSignalOfCounters,


   CFG_FeaturePoGateControlOfCounters,
   CFG_FeaturePoGatePolarities,
   CFG_FeaturePoHiPeriodRange,
   CFG_FeaturePoLoPeriodRange,
   CFG_FeaturePoOutCountRange,

   CFG_PoGateEnabledOfCounters,
   CFG_PoGatePolarityOfCounters,
   CFG_PoOutCountOfCounters,




   CFG_FeatureChipClkPolarities,
   CFG_FeatureChipGatePolarities,
   CFG_FeatureChipOutSignals,

   CFG_ChipClkPolarityOfCounters,
   CFG_ChipGatePolarityOfCounters,
   CFG_ChipOutSignalOfCounters,




   CFG_FeatureOsDelayCountRange,




   CFG_FeatureUdCountingTypes,
   CFG_FeatureUdInitialValues,
   CFG_UdCountingTypeOfCounters,
   CFG_UdInitialValueOfCounters,
   CFG_UdCountValueResetTimesByIndexs,




   CFG_FeatureFilterTypes,
   CFG_FeatureFilterCutoffFreqRange,
   CFG_FeatureFilterCutoffFreq1Range,
   CFG_FilterTypeOfChannels,
   CFG_FilterCutoffFreqOfChannels,
   CFG_FilterCutoffFreq1OfChannels,




   CFG_FeatureDiOpenStatePorts,
   CFG_FeatureDiOpenStates,
   CFG_DiOpenStatesOfPorts,




   CFG_FeaturePoOutSignals,
   CFG_PoOutSignalOfCounters,




   CFG_FretureTriggerSourceVRG,
   CFG_FretureTriggerHysteresisIndexMax,
   CFG_FretureTriggerHysteresisIndexStep,
   CFG_TriggerHysteresisIndex,
   CFG_FretureTriggerSourceVRG1,
   CFG_FretureTriggerHysteresisIndexMax1,
   CFG_FretureTriggerHysteresisIndexStep1,
   CFG_TriggerHysteresisIndex1,



   CFG_FeatureCouplingType,
   CFG_CouplingTypeOfChannels,
   CFG_FeatureImpedanceType,
   CFG_ImpedanceTypeOfChannels,
   CFG_FaiRecordCount,

   CFG_FretureTriggerSourceFilterTypes,
   CFG_FretureTriggerSourceFilterTypes1,
   CFG_TriggerSourceFilterTypes,
   CFG_TriggerSourceFilterTypes1,

} PropertyId;



typedef enum tagErrorCode {



   Success = 0,







   WarningIntrNotAvailable = 0xA0000000,




   WarningParamOutOfRange = 0xA0000001,




   WarningPropValueOutOfRange = 0xA0000002,




   WarningPropValueNotSpted = 0xA0000003,




   WarningPropValueConflict = 0xA0000004,





   WarningVrgOfGroupNotSame = 0xA0000005,







   ErrorHandleNotValid = 0xE0000000,




   ErrorParamOutOfRange = 0xE0000001,




   ErrorParamNotSpted = 0xE0000002,




   ErrorParamFmtUnexpted = 0xE0000003,




   ErrorMemoryNotEnough = 0xE0000004,




   ErrorBufferIsNull = 0xE0000005,




   ErrorBufferTooSmall = 0xE0000006,




   ErrorDataLenExceedLimit = 0xE0000007,




   ErrorFuncNotSpted = 0xE0000008,




   ErrorEventNotSpted = 0xE0000009,




   ErrorPropNotSpted = 0xE000000A,




   ErrorPropReadOnly = 0xE000000B,




   ErrorPropValueConflict = 0xE000000C,




   ErrorPropValueOutOfRange = 0xE000000D,




   ErrorPropValueNotSpted = 0xE000000E,




   ErrorPrivilegeNotHeld = 0xE000000F,




   ErrorPrivilegeNotAvailable = 0xE0000010,




   ErrorDriverNotFound = 0xE0000011,




   ErrorDriverVerMismatch = 0xE0000012,




   ErrorDriverCountExceedLimit = 0xE0000013,




   ErrorDeviceNotOpened = 0xE0000014,




   ErrorDeviceNotExist = 0xE0000015,




   ErrorDeviceUnrecognized = 0xE0000016,




   ErrorConfigDataLost = 0xE0000017,




   ErrorFuncNotInited = 0xE0000018,




   ErrorFuncBusy = 0xE0000019,




   ErrorIntrNotAvailable = 0xE000001A,




   ErrorDmaNotAvailable = 0xE000001B,




   ErrorDeviceIoTimeOut = 0xE000001C,




   ErrorSignatureNotMatch = 0xE000001D,




   ErrorFuncConflictWithBfdAi = 0xE000001E,




   ErrorVrgNotAvailableInSeMode = 0xE000001F,




   ErrorUndefined = 0xE000FFFF,
} ErrorCode;


typedef enum tagProductId {
   BD_DEMO = 0x00,
   BD_PCL818 = 0x05,
   BD_PCL818H = 0x11,
   BD_PCL818L = 0x21,
   BD_PCL818HG = 0x22,
   BD_PCL818HD = 0x2b,
   BD_PCM3718 = 0x37,
   BD_PCM3724 = 0x38,
   BD_PCM3730 = 0x5a,
   BD_PCI1750 = 0x5e,
   BD_PCI1751 = 0x5f,
   BD_PCI1710 = 0x60,
   BD_PCI1712 = 0x61,
   BD_PCI1710HG = 0x67,
   BD_PCI1711 = 0x73,
   BD_PCI1711L = 0x75,
   BD_PCI1713 = 0x68,
   BD_PCI1753 = 0x69,
   BD_PCI1760 = 0x6a,
   BD_PCI1720 = 0x6b,
   BD_PCM3718H = 0x6d,
   BD_PCM3718HG = 0x6e,
   BD_PCI1716 = 0x74,
   BD_PCI1731 = 0x75,
   BD_PCI1754 = 0x7b,
   BD_PCI1752 = 0x7c,
   BD_PCI1756 = 0x7d,
   BD_PCM3725 = 0x7f,
   BD_PCI1762 = 0x80,
   BD_PCI1721 = 0x81,
   BD_PCI1761 = 0x82,
   BD_PCI1723 = 0x83,
   BD_PCI1730 = 0x87,
   BD_PCI1733 = 0x88,
   BD_PCI1734 = 0x89,
   BD_PCI1710L = 0x90,
   BD_PCI1710HGL = 0x91,
   BD_PCM3712 = 0x93,
   BD_PCM3723 = 0x94,
   BD_PCI1780 = 0x95,
   BD_MIC3756 = 0x96,
   BD_PCI1755 = 0x97,
   BD_PCI1714 = 0x98,
   BD_PCI1757 = 0x99,
   BD_MIC3716 = 0x9A,
   BD_MIC3761 = 0x9B,
   BD_MIC3753 = 0x9C,
   BD_MIC3780 = 0x9D,
   BD_PCI1724 = 0x9E,
   BD_PCI1758UDI = 0xA3,
   BD_PCI1758UDO = 0xA4,
   BD_PCI1747 = 0xA5,
   BD_PCM3780 = 0xA6,
   BD_MIC3747 = 0xA7,
   BD_PCI1758UDIO = 0xA8,
   BD_PCI1712L = 0xA9,
   BD_PCI1763UP = 0xAC,
   BD_PCI1736UP = 0xAD,
   BD_PCI1714UL = 0xAE,
   BD_MIC3714 = 0xAF,
   BD_PCM3718HO = 0xB1,
   BD_PCI1741U = 0xB3,
   BD_MIC3723 = 0xB4,
   BD_PCI1718HDU = 0xB5,
   BD_MIC3758DIO = 0xB6,
   BD_PCI1727U = 0xB7,
   BD_PCI1718HGU = 0xB8,
   BD_PCI1715U = 0xB9,
   BD_PCI1716L = 0xBA,
   BD_PCI1735U = 0xBB,
   BD_USB4711 = 0xBC,
   BD_PCI1737U = 0xBD,
   BD_PCI1739U = 0xBE,
   BD_PCI1742U = 0xC0,
   BD_USB4718 = 0xC6,
   BD_MIC3755 = 0xC7,
   BD_USB4761 = 0xC8,
   BD_PCI1784 = 0XCC,
   BD_USB4716 = 0xCD,
   BD_PCI1752U = 0xCE,
   BD_PCI1752USO = 0xCF,
   BD_USB4751 = 0xD0,
   BD_USB4751L = 0xD1,
   BD_USB4750 = 0xD2,
   BD_MIC3713 = 0xD3,
   BD_USB4711A = 0xD8,
   BD_PCM3753P = 0xD9,
   BD_PCM3784 = 0xDA,
   BD_PCM3761I = 0xDB,
   BD_MIC3751 = 0xDC,
   BD_PCM3730I = 0xDD,
   BD_PCM3813I = 0xE0,
   BD_PCIE1744 = 0xE1,
   BD_PCI1730U = 0xE2,
   BD_PCI1760U = 0xE3,
   BD_MIC3720 = 0xE4,
   BD_PCM3810I = 0xE9,
   BD_USB4702 = 0xEA,
   BD_USB4704 = 0xEB,
   BD_PCM3810I_HG = 0xEC,
   BD_PCI1713U = 0xED,


   BD_PCI1706U = 0x800,
   BD_PCI1706MSU = 0x801,
   BD_PCI1706UL = 0x802,
   BD_PCIE1752 = 0x803,
   BD_PCIE1754 = 0x804,
   BD_PCIE1756 = 0x805,
   BD_MIC1911 = 0x806,
   BD_MIC3750 = 0x807,
   BD_MIC3711 = 0x808,
   BD_PCIE1730 = 0x809,
   BD_PCI1710_ECU = 0x80A,
   BD_PCI1720_ECU = 0x80B,
   BD_PCIE1760 = 0x80C,
   BD_PCIE1751 = 0x80D,
   BD_ECUP1706 = 0x80E,
   BD_PCIE1753 = 0x80F,
   BD_PCIE1810 = 0x810,
   BD_ECUP1702L = 0x811,
   BD_PCIE1816 = 0x812,
   BD_PCM27D24DI = 0x813,
   BD_PCIE1816H = 0x814,
   BD_PCIE1840 = 0x815,
   BD_PCL725 = 0x816,
} ProductId;


# 1518 "bdaqctrl.h"





typedef struct tagDeviceInformation{
   int32 DeviceNumber;
   AccessMode DeviceMode;
   int32 ModuleIndex;
   wchar_t Description[64];
# 1550 "bdaqctrl.h"
} DeviceInformation;

typedef struct tagDeviceTreeNode{
   int32 DeviceNumber;
   int32 ModulesIndex[8];
   wchar_t Description[64];
}DeviceTreeNode;

typedef struct tagDeviceEventArgs {



   int32 dummy[1];
}DeviceEventArgs;

typedef struct tagBfdAiEventArgs {
   int32 Offset;
   int32 Count;
}BfdAiEventArgs;

typedef struct tagBfdAoEventArgs {
   int32 Offset;
   int32 Count;
}BfdAoEventArgs;

typedef struct tagBfdDiEventArgs {
   int32 Offset;
   int32 Count;
}BfdDiEventArgs;

typedef struct tagBfdDoEventArgs {
   int32 Offset;
   int32 Count;
}BfdDoEventArgs;

typedef struct tagDiSnapEventArgs{
   int32 SrcNum;
   int32 Length;
   uint8 PortData[32];
}DiSnapEventArgs;

typedef struct tagCntrEventArgs{
   int32 Channel;
}CntrEventArgs;

typedef struct tagUdCntrEventArgs{
   int32 SrcId;
   int32 Length;
   int32 Data[8];
}UdCntrEventArgs;

typedef struct tagPulseWidth{
   double HiPeriod;
   double LoPeriod;
}PulseWidth;

typedef enum tagControlState
{
   Idle = 0,
   Ready,
   Running,
   Stopped
} ControlState;
# 2618 "bdaqctrl.h"
   typedef struct ICollection ICollection;
   typedef struct AnalogChannel AnalogChannel;
   typedef struct AnalogInputChannel AnalogInputChannel;
   typedef struct CjcSetting CjcSetting;
   typedef struct ScanChannel ScanChannel;
   typedef struct ConvertClock ConvertClock;
   typedef struct ScanClock ScanClock;
   typedef struct Trigger Trigger;
   typedef struct PortDirection PortDirection;
   typedef struct NoiseFilterChannel NoiseFilterChannel;
   typedef struct DiintChannel DiintChannel;
   typedef struct DiCosintPort DiCosintPort;
   typedef struct DiPmintPort DiPmintPort;
   typedef struct ScanPort ScanPort;

   typedef struct DeviceEventHandler {
      void ( *DeviceEvent)(void *_this, void *sender, DeviceEventArgs *args);
   } DeviceEventHandler;

   typedef struct DeviceEventListener {
      DeviceEventHandler const *vtbl;
   }DeviceEventListener;




   typedef struct BfdAiEventHandler {
      void ( *BfdAiEvent)(void *_this, void *sender, BfdAiEventArgs *args);
   } BfdAiEventHandler;

   typedef struct BfdAiEventListener {
      BfdAiEventHandler const *vtbl;
   } BfdAiEventListener;

   typedef struct AiFeatures AiFeatures;
   typedef struct InstantAiCtrl InstantAiCtrl;
   typedef struct BufferedAiCtrl BufferedAiCtrl;




   typedef struct BfdAoEventHandler {
      void ( *BfdAoEvent)(void *_this, void *sender, BfdAoEventArgs *args);
   } BfdAoEventHandler;

   typedef struct BfdAoEventListener {
      BfdAoEventHandler const *vtbl;
   } BfdAoEventListener;

   typedef struct AoFeatures AoFeatures;
   typedef struct InstantAoCtrl InstantAoCtrl;
   typedef struct BufferedAoCtrl BufferedAoCtrl;




   typedef struct DiSnapEventHandler {
      void ( *DiSnapEvent)(void *_this, void *sender, DiSnapEventArgs *args);
   } DiSnapEventHandler;

   typedef struct DiSnapEventListener {
      DiSnapEventHandler const *vtbl;
   } DiSnapEventListener;

   typedef struct BfdDiEventHandler {
      void ( *BfdDiEvent)(void *_this, void *sender, BfdDiEventArgs *args);
   } BfdDiEventHandler;

   typedef struct BfdDiEventListener {
      BfdDiEventHandler const *vtbl;
   } BfdDiEventListener;

   typedef struct DiFeatures DiFeatures;
   typedef struct InstantDiCtrl InstantDiCtrl;
   typedef struct InstantDoCtrl InstantDoCtrl;

   typedef struct BfdDoEventHandler {
      void ( *BfdDoEvent)(void *_this, void *sender, BfdDoEventArgs *args);
   } BfdDoEventHandler;

   typedef struct BfdDoEventListener {
      BfdDoEventHandler const *vtbl;
   } BfdDoEventListener;

   typedef struct DoFeatures DoFeatures;
   typedef struct BufferedDiCtrl BufferedDiCtrl;
   typedef struct BufferedDoCtrl BufferedDoCtrl;




   typedef struct CntrEventHandler {
      void ( *CntrEvent)(void *_this, void *sender, CntrEventArgs *args);
   } CntrEventHandler;

   typedef struct CntrEventListener {
      CntrEventHandler const *vtbl;
   } CntrEventListener;

   typedef struct CounterCapabilityIndexer CounterCapabilityIndexer;

   typedef struct EventCounterFeatures EventCounterFeatures;
   typedef struct EventCounterCtrl EventCounterCtrl;

   typedef struct FreqMeterFeatures FreqMeterFeatures;
   typedef struct FreqMeterCtrl FreqMeterCtrl;

   typedef struct OneShotFeatures OneShotFeatures;
   typedef struct OneShotCtrl OneShotCtrl;

   typedef struct TimerPulseFeatures TimerPulseFeatures;
   typedef struct TimerPulseCtrl TimerPulseCtrl;

   typedef struct PwMeterFeatures PwMeterFeatures;
   typedef struct PwMeterCtrl PwMeterCtrl;

   typedef struct PwModulatorFeatures PwModulatorFeatures;
   typedef struct PwModulatorCtrl PwModulatorCtrl;

   typedef struct UdCntrEventHandler {
      void ( *UdCntrEvent)(void *_this, void *sender, UdCntrEventArgs *args);
   } UdCntrEventHandler;

   typedef struct UdCntrEventListener {
      UdCntrEventHandler const *vtbl;
   } UdCntrEventListener;

   typedef struct UdCounterFeatures UdCounterFeatures;
   typedef struct UdCounterCtrl UdCounterCtrl;
# 3968 "bdaqctrl.h"
         ErrorCode AdxDeviceGetLinkageInfo(
            int32 deviceParent,
            int32 index,
            int32 *deviceNumber,
            wchar_t *description,
            int32 *subDeviceCount);

         ErrorCode AdxGetValueRangeInformation(
            ValueRange type,
            int32 descBufSize,
            wchar_t *description,
            MathInterval *range,
            ValueUnit *unit);

         ErrorCode AdxGetSignalConnectionInformation(
            SignalDrop signal,
            int32 descBufSize,
            wchar_t *description,
            SignalPosition *position);

         ErrorCode AdxEnumToString(
            wchar_t const *enumTypeName,
            int32 enumValue,
            int32 enumStringLength,
            wchar_t *enumString);

         ErrorCode AdxStringToEnum(
            wchar_t const *enumTypeName,
            wchar_t const *enumString,
            int32 *enumValue);


         InstantAiCtrl* AdxInstantAiCtrlCreate();

         BufferedAiCtrl* AdxBufferedAiCtrlCreate();

         InstantAoCtrl* AdxInstantAoCtrlCreate();

         BufferedAoCtrl* AdxBufferedAoCtrlCreate();

         InstantDiCtrl* AdxInstantDiCtrlCreate();

         BufferedDiCtrl* AdxBufferedDiCtrlCreate();

         InstantDoCtrl* AdxInstantDoCtrlCreate();

         BufferedDoCtrl* AdxBufferedDoCtrlCreate();

         EventCounterCtrl* AdxEventCounterCtrlCreate();

         FreqMeterCtrl* AdxFreqMeterCtrlCreate();

         OneShotCtrl* AdxOneShotCtrlCreate();

         PwMeterCtrl* AdxPwMeterCtrlCreate();

         PwModulatorCtrl* AdxPwModulatorCtrlCreate();

         TimerPulseCtrl* AdxTimerPulseCtrlCreate();

         UdCounterCtrl* AdxUdCounterCtrlCreate();






         void ICollection_Dispose(ICollection *_this);
         int32 ICollection_getCount(ICollection *_this);
         void* ICollection_getItem(ICollection *_this, int32 index);




         int32 AnalogChannel_getChannel(AnalogChannel *_this);
         ValueRange AnalogChannel_getValueRange(AnalogChannel *_this);
         ErrorCode AnalogChannel_setValueRange(AnalogChannel *_this, ValueRange value);




         int32 AnalogInputChannel_getChannel(AnalogInputChannel *_this);
         ValueRange AnalogInputChannel_getValueRange(AnalogInputChannel *_this);
         ErrorCode AnalogInputChannel_setValueRange(AnalogInputChannel *_this, ValueRange value);
         AiSignalType AnalogInputChannel_getSignalType(AnalogInputChannel *_this);
         ErrorCode AnalogInputChannel_setSignalType(AnalogInputChannel *_this, AiSignalType value);
         BurnoutRetType AnalogInputChannel_getBurnoutRetType(AnalogInputChannel *_this);
         ErrorCode AnalogInputChannel_setBurnoutRetType(AnalogInputChannel *_this, BurnoutRetType value);
         double AnalogInputChannel_getBurnoutRetValue(AnalogInputChannel *_this);
         ErrorCode AnalogInputChannel_setBurnoutRetValue(AnalogInputChannel *_this, double value);




         int32 CjcSetting_getChannel(CjcSetting *_this);
         ErrorCode CjcSetting_setChannel(CjcSetting *_this, int32 ch);
         double CjcSetting_getValue(CjcSetting *_this);
         ErrorCode CjcSetting_setValue(CjcSetting *_this, double value);




         int32 ScanChannel_getChannelStart(ScanChannel *_this);
         ErrorCode ScanChannel_setChannelStart(ScanChannel *_this, int32 value);
         int32 ScanChannel_getChannelCount(ScanChannel *_this);
         ErrorCode ScanChannel_setChannelCount(ScanChannel *_this, int32 value);
         int32 ScanChannel_getSamples(ScanChannel *_this);
         ErrorCode ScanChannel_setSamples(ScanChannel *_this, int32 value);
         int32 ScanChannel_getIntervalCount(ScanChannel *_this);
         ErrorCode ScanChannel_setIntervalCount(ScanChannel *_this, int32 value);




         SignalDrop ConvertClock_getSource(ConvertClock *_this);
         ErrorCode ConvertClock_setSource(ConvertClock *_this, SignalDrop value);
         double ConvertClock_getRate(ConvertClock *_this);
         ErrorCode ConvertClock_setRate(ConvertClock *_this, double value);




         SignalDrop ScanClock_getSource(ScanClock *_this);
         ErrorCode ScanClock_setSource(ScanClock *_this, SignalDrop value);
         double ScanClock_getRate(ScanClock *_this);
         ErrorCode ScanClock_setRate(ScanClock *_this, double value);
         int32 ScanClock_getScanCount(ScanClock *_this);
         ErrorCode ScanClock_setScanCount(ScanClock *_this, int32 value);




         SignalDrop Trigger_getSource(Trigger *_this);
         ErrorCode Trigger_setSource(Trigger *_this,SignalDrop value);
         ActiveSignal Trigger_getEdge(Trigger *_this);
         ErrorCode Trigger_setEdge(Trigger *_this, ActiveSignal value);
         double Trigger_getLevel(Trigger *_this);
         ErrorCode Trigger_setLevel(Trigger *_this, double value);
         TriggerAction Trigger_getAction(Trigger *_this);
         ErrorCode Trigger_setAction(Trigger *_this, TriggerAction value);
         int32 Trigger_getDelayCount(Trigger *_this);
         ErrorCode Trigger_setDelayCount(Trigger *_this, int32 value);




         int32 PortDirection_getPort(PortDirection *_this);
         DioPortDir PortDirection_getDirection(PortDirection *_this);
         ErrorCode PortDirection_setDirection(PortDirection *_this, DioPortDir value);




         int32 NoiseFilterChannel_getChannel(NoiseFilterChannel *_this);
         int8 NoiseFilterChannel_getEnabled(NoiseFilterChannel *_this);
         ErrorCode NoiseFilterChannel_setEnabled(NoiseFilterChannel *_this, int8 value);




         int32 DiintChannel_getChannel(DiintChannel *_this);
         int8 DiintChannel_getEnabled(DiintChannel *_this);
         ErrorCode DiintChannel_setEnabled(DiintChannel *_this, int8 value);
         int8 DiintChannel_getGated(DiintChannel *_this);
         ErrorCode DiintChannel_setGated(DiintChannel *_this, int8 value);
         ActiveSignal DiintChannel_getTrigEdge(DiintChannel *_this);
         ErrorCode DiintChannel_setTrigEdge(DiintChannel *_this, ActiveSignal value);




         int32 DiCosintPort_getPort(DiCosintPort *_this);
         uint8 DiCosintPort_getMask(DiCosintPort *_this);
         ErrorCode DiCosintPort_setMask(DiCosintPort *_this, uint8 value);




         int32 DiPmintPort_getPort(DiPmintPort *_this);
         uint8 DiPmintPort_getMask(DiPmintPort *_this);
         ErrorCode DiPmintPort_setMask(DiPmintPort *_this, uint8 value);
         uint8 DiPmintPort_getPattern(DiPmintPort *_this);
         ErrorCode DiPmintPort_setPattern(DiPmintPort *_this, uint8 value);




         int32 ScanPort_getPortStart(ScanPort *_this);
         ErrorCode ScanPort_setPortStart(ScanPort *_this, int32 value);
         int32 ScanPort_getPortCount(ScanPort *_this);
         ErrorCode ScanPort_setPortCount(ScanPort *_this, int32 value);
         int32 ScanPort_getSamples(ScanPort *_this);
         ErrorCode ScanPort_setSamples(ScanPort *_this, int32 value);
         int32 ScanPort_getIntervalCount(ScanPort *_this);
         ErrorCode ScanPort_setIntervalCount(ScanPort *_this, int32 value);





         int32 AiFeatures_getResolution(AiFeatures *_this);
         int32 AiFeatures_getDataSize(AiFeatures *_this);
         int32 AiFeatures_getDataMask(AiFeatures *_this);

         int32 AiFeatures_getChannelCountMax(AiFeatures *_this);
         AiChannelType AiFeatures_getChannelType(AiFeatures *_this);
         int8 AiFeatures_getOverallValueRange(AiFeatures *_this);
         int8 AiFeatures_getThermoSupported(AiFeatures *_this);
         ICollection* AiFeatures_getValueRanges(AiFeatures *_this);
         ICollection* AiFeatures_getBurnoutReturnTypes(AiFeatures *_this);

         ICollection* AiFeatures_getCjcChannels(AiFeatures *_this);

         int8 AiFeatures_getBufferedAiSupported(AiFeatures *_this);
         SamplingMethod AiFeatures_getSamplingMethod(AiFeatures *_this);
         int32 AiFeatures_getChannelStartBase(AiFeatures *_this);
         int32 AiFeatures_getChannelCountBase(AiFeatures *_this);

         ICollection* AiFeatures_getConvertClockSources(AiFeatures *_this);
         void AiFeatures_getConvertClockRange(AiFeatures *_this, MathInterval *value);

         int8 AiFeatures_getBurstScanSupported(AiFeatures *_this);
         ICollection* AiFeatures_getScanClockSources(AiFeatures *_this);
         void AiFeatures_getScanClockRange(AiFeatures *_this, MathInterval *value);
         int32 AiFeatures_getScanCountMax(AiFeatures *_this);

         int8 AiFeatures_getTriggerSupported(AiFeatures *_this);
         int32 AiFeatures_getTriggerCount(AiFeatures *_this);
         ICollection* AiFeatures_getTriggerSources(AiFeatures *_this);
         ICollection* AiFeatures_getTriggerActions(AiFeatures *_this);
         void AiFeatures_getTriggerDelayRange(AiFeatures *_this, MathInterval *value);

         int8 AiFeatures_getTrigger1Supported(AiFeatures *_this);
         ICollection* AiFeatures_getTrigger1Sources(AiFeatures *_this);
         ICollection* AiFeatures_getTrigger1Actions(AiFeatures *_this);
         void AiFeatures_getTrigger1DelayRange(AiFeatures *_this, MathInterval *value);





         void InstantAiCtrl_Dispose(InstantAiCtrl *_this);
         void InstantAiCtrl_Cleanup(InstantAiCtrl *_this);
         ErrorCode InstantAiCtrl_UpdateProperties(InstantAiCtrl *_this);
         void InstantAiCtrl_addRemovedListener(InstantAiCtrl *_this, DeviceEventListener * listener);
         void InstantAiCtrl_removeRemovedListener(InstantAiCtrl *_this, DeviceEventListener * listener);
         void InstantAiCtrl_addReconnectedListener(InstantAiCtrl *_this, DeviceEventListener * listener);
         void InstantAiCtrl_removeReconnectedListener(InstantAiCtrl *_this, DeviceEventListener * listener);
         void InstantAiCtrl_addPropertyChangedListener(InstantAiCtrl *_this, DeviceEventListener * listener);
         void InstantAiCtrl_removePropertyChangedListener(InstantAiCtrl *_this, DeviceEventListener * listener);
         void InstantAiCtrl_getSelectedDevice(InstantAiCtrl *_this, DeviceInformation *x);
         ErrorCode InstantAiCtrl_setSelectedDevice(InstantAiCtrl *_this, DeviceInformation const *x);
         int8 InstantAiCtrl_getInitialized(InstantAiCtrl *_this);
         int8 InstantAiCtrl_getCanEditProperty(InstantAiCtrl *_this);
         HANDLE InstantAiCtrl_getDevice(InstantAiCtrl *_this);
         HANDLE InstantAiCtrl_getModule(InstantAiCtrl *_this);
         ICollection* InstantAiCtrl_getSupportedDevices(InstantAiCtrl *_this);
         ICollection* InstantAiCtrl_getSupportedModes(InstantAiCtrl *_this);

         AiFeatures* InstantAiCtrl_getFeatures(InstantAiCtrl *_this);
         ICollection* InstantAiCtrl_getChannels(InstantAiCtrl *_this);
         int32 InstantAiCtrl_getChannelCount(InstantAiCtrl *_this);

         ErrorCode InstantAiCtrl_ReadAny(InstantAiCtrl *_this, int32 chStart, int32 chCount, void *dataRaw, double *dataScaled);
         CjcSetting* InstantAiCtrl_getCjc(InstantAiCtrl *_this);





         void BufferedAiCtrl_Dispose(BufferedAiCtrl *_this);
         void BufferedAiCtrl_Cleanup(BufferedAiCtrl *_this);
         ErrorCode BufferedAiCtrl_UpdateProperties(BufferedAiCtrl *_this);
         void BufferedAiCtrl_addRemovedListener(BufferedAiCtrl *_this, DeviceEventListener * listener);
         void BufferedAiCtrl_removeRemovedListener(BufferedAiCtrl *_this, DeviceEventListener * listener);
         void BufferedAiCtrl_addReconnectedListener(BufferedAiCtrl *_this, DeviceEventListener * listener);
         void BufferedAiCtrl_removeReconnectedListener(BufferedAiCtrl *_this, DeviceEventListener * listener);
         void BufferedAiCtrl_addPropertyChangedListener(BufferedAiCtrl *_this, DeviceEventListener * listener);
         void BufferedAiCtrl_removePropertyChangedListener(BufferedAiCtrl *_this, DeviceEventListener * listener);
         void BufferedAiCtrl_getSelectedDevice(BufferedAiCtrl *_this, DeviceInformation *x);
         ErrorCode BufferedAiCtrl_setSelectedDevice(BufferedAiCtrl *_this, DeviceInformation const *x);
         int8 BufferedAiCtrl_getInitialized(BufferedAiCtrl *_this);
         int8 BufferedAiCtrl_getCanEditProperty(BufferedAiCtrl *_this);
         HANDLE BufferedAiCtrl_getDevice(BufferedAiCtrl *_this);
         HANDLE BufferedAiCtrl_getModule(BufferedAiCtrl *_this);
         ICollection* BufferedAiCtrl_getSupportedDevices(BufferedAiCtrl *_this);
         ICollection* BufferedAiCtrl_getSupportedModes(BufferedAiCtrl *_this);

         AiFeatures* BufferedAiCtrl_getFeatures(BufferedAiCtrl *_this);
         ICollection* BufferedAiCtrl_getChannels(BufferedAiCtrl *_this);
         int32 BufferedAiCtrl_getChannelCount(BufferedAiCtrl *_this);


         void BufferedAiCtrl_addDataReadyListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);
         void BufferedAiCtrl_removeDataReadyListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);
         void BufferedAiCtrl_addOverrunListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);
         void BufferedAiCtrl_removeOverrunListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);
         void BufferedAiCtrl_addCacheOverflowListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);
         void BufferedAiCtrl_removeCacheOverflowListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);
         void BufferedAiCtrl_addStoppedListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);
         void BufferedAiCtrl_removeStoppedListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);

         ErrorCode BufferedAiCtrl_Prepare(BufferedAiCtrl *_this);
         ErrorCode BufferedAiCtrl_RunOnce(BufferedAiCtrl *_this);
         ErrorCode BufferedAiCtrl_Start(BufferedAiCtrl *_this);
         ErrorCode BufferedAiCtrl_Stop(BufferedAiCtrl *_this);
         ErrorCode BufferedAiCtrl_GetDataI16(BufferedAiCtrl *_this, int32 count, int16 rawData[]);
         ErrorCode BufferedAiCtrl_GetDataI32(BufferedAiCtrl *_this, int32 count, int32 rawData[]);
         ErrorCode BufferedAiCtrl_GetDataF64(BufferedAiCtrl *_this, int32 count, double scaledData[]);
         void BufferedAiCtrl_Release(BufferedAiCtrl *_this);

         void* BufferedAiCtrl_getBuffer(BufferedAiCtrl *_this);
         int32 BufferedAiCtrl_getBufferCapacity(BufferedAiCtrl *_this);
         ControlState BufferedAiCtrl_getState(BufferedAiCtrl *_this);
         ScanChannel* BufferedAiCtrl_getScanChannel(BufferedAiCtrl *_this);
         ConvertClock* BufferedAiCtrl_getConvertClock(BufferedAiCtrl *_this);
         ScanClock* BufferedAiCtrl_getScanClock(BufferedAiCtrl *_this);
         Trigger* BufferedAiCtrl_getTrigger(BufferedAiCtrl *_this);
         int8 BufferedAiCtrl_getStreaming(BufferedAiCtrl *_this);
         ErrorCode BufferedAiCtrl_setStreaming(BufferedAiCtrl *_this, int8 value);

         ErrorCode BufferedAiCtrl_GetEventStatus(BufferedAiCtrl *_this, EventId id, int32 *status);

         Trigger* BufferedAiCtrl_getTrigger1(BufferedAiCtrl *_this);

         void BufferedAiCtrl_addRecordReadyListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);
         void BufferedAiCtrl_removeRecordReadyListener(BufferedAiCtrl *_this, BfdAiEventListener *listener);





         int32 AoFeatures_getResolution(AoFeatures *_this);
         int32 AoFeatures_getDataSize(AoFeatures *_this);
         int32 AoFeatures_getDataMask(AoFeatures *_this);

         int32 AoFeatures_getChannelCountMax(AoFeatures *_this);
         ICollection* AoFeatures_getValueRanges(AoFeatures *_this);
         int8 AoFeatures_getExternalRefAntiPolar(AoFeatures *_this);
         void AoFeatures_getExternalRefRange(AoFeatures *_this, MathInterval *value);

         int8 AoFeatures_getBufferedAoSupported(AoFeatures *_this);
         SamplingMethod AoFeatures_getSamplingMethod(AoFeatures *_this);
         int32 AoFeatures_getChannelStartBase(AoFeatures *_this);
         int32 AoFeatures_getChannelCountBase(AoFeatures *_this);

         ICollection* AoFeatures_getConvertClockSources(AoFeatures *_this);
         void AoFeatures_getConvertClockRange(AoFeatures *_this, MathInterval *value);

         int8 AoFeatures_getTriggerSupported(AoFeatures *_this);
         int32 AoFeatures_getTriggerCount(AoFeatures *_this);
         ICollection* AoFeatures_getTriggerSources(AoFeatures *_this);
         ICollection* AoFeatures_getTriggerActions(AoFeatures *_this);
         void AoFeatures_getTriggerDelayRange(AoFeatures *_this, MathInterval *value);

         int8 AoFeatures_getTrigger1Supported(AoFeatures *_this);
         ICollection* AoFeatures_getTrigger1Sources(AoFeatures *_this);
         ICollection* AoFeatures_getTrigger1Actions(AoFeatures *_this);
         MathInterval AoFeatures_getTrigger1DelayRange(AoFeatures *_this);





         void InstantAoCtrl_Dispose(InstantAoCtrl *_this);
         void InstantAoCtrl_Cleanup(InstantAoCtrl *_this);
         ErrorCode InstantAoCtrl_UpdateProperties(InstantAoCtrl *_this);
         void InstantAoCtrl_addRemovedListener(InstantAoCtrl *_this, DeviceEventListener * listener);
         void InstantAoCtrl_removeRemovedListener(InstantAoCtrl *_this, DeviceEventListener * listener);
         void InstantAoCtrl_addReconnectedListener(InstantAoCtrl *_this, DeviceEventListener * listener);
         void InstantAoCtrl_removeReconnectedListener(InstantAoCtrl *_this, DeviceEventListener * listener);
         void InstantAoCtrl_addPropertyChangedListener(InstantAoCtrl *_this, DeviceEventListener * listener);
         void InstantAoCtrl_removePropertyChangedListener(InstantAoCtrl *_this, DeviceEventListener * listener);
         void InstantAoCtrl_getSelectedDevice(InstantAoCtrl *_this, DeviceInformation *x);
         ErrorCode InstantAoCtrl_setSelectedDevice(InstantAoCtrl *_this, DeviceInformation const *x);
         int8 InstantAoCtrl_getInitialized(InstantAoCtrl *_this);
         int8 InstantAoCtrl_getCanEditProperty(InstantAoCtrl *_this);
         HANDLE InstantAoCtrl_getDevice(InstantAoCtrl *_this);
         HANDLE InstantAoCtrl_getModule(InstantAoCtrl *_this);
         ICollection* InstantAoCtrl_getSupportedDevices(InstantAoCtrl *_this);
         ICollection* InstantAoCtrl_getSupportedModes(InstantAoCtrl *_this);

         AoFeatures* InstantAoCtrl_getFeatures(InstantAoCtrl *_this);
         ICollection* InstantAoCtrl_getChannels(InstantAoCtrl *_this);
         int32 InstantAoCtrl_getChannelCount(InstantAoCtrl *_this);
         double InstantAoCtrl_getExtRefValueForUnipolar(InstantAoCtrl *_this);
         ErrorCode InstantAoCtrl_setExtRefValueForUnipolar(InstantAoCtrl *_this, double value);
         double InstantAoCtrl_getExtRefValueForBipolar(InstantAoCtrl *_this);
         ErrorCode InstantAoCtrl_setExtRefValueForBipolar(InstantAoCtrl *_this, double value);

         ErrorCode InstantAoCtrl_WriteAny(InstantAoCtrl *_this, int32 chStart, int32 chCount, void *dataRaw, double *dataScaled);





         void BufferedAoCtrl_Dispose(BufferedAoCtrl *_this);
         void BufferedAoCtrl_Cleanup(BufferedAoCtrl *_this);
         ErrorCode BufferedAoCtrl_UpdateProperties(BufferedAoCtrl *_this);
         void BufferedAoCtrl_addRemovedListener(BufferedAoCtrl *_this, DeviceEventListener * listener);
         void BufferedAoCtrl_removeRemovedListener(BufferedAoCtrl *_this, DeviceEventListener * listener);
         void BufferedAoCtrl_addReconnectedListener(BufferedAoCtrl *_this, DeviceEventListener * listener);
         void BufferedAoCtrl_removeReconnectedListener(BufferedAoCtrl *_this, DeviceEventListener * listener);
         void BufferedAoCtrl_addPropertyChangedListener(BufferedAoCtrl *_this, DeviceEventListener * listener);
         void BufferedAoCtrl_removePropertyChangedListener(BufferedAoCtrl *_this, DeviceEventListener * listener);
         void BufferedAoCtrl_getSelectedDevice(BufferedAoCtrl *_this, DeviceInformation *x);
         ErrorCode BufferedAoCtrl_setSelectedDevice(BufferedAoCtrl *_this, DeviceInformation const *x);
         int8 BufferedAoCtrl_getInitialized(BufferedAoCtrl *_this);
         int8 BufferedAoCtrl_getCanEditProperty(BufferedAoCtrl *_this);
         HANDLE BufferedAoCtrl_getDevice(BufferedAoCtrl *_this);
         HANDLE BufferedAoCtrl_getModule(BufferedAoCtrl *_this);
         ICollection* BufferedAoCtrl_getSupportedDevices(BufferedAoCtrl *_this);
         ICollection* BufferedAoCtrl_getSupportedModes(BufferedAoCtrl *_this);

         AoFeatures* BufferedAoCtrl_getFeatures(BufferedAoCtrl *_this);
         ICollection* BufferedAoCtrl_getChannels(BufferedAoCtrl *_this);
         int32 BufferedAoCtrl_getChannelCount(BufferedAoCtrl *_this);
         double BufferedAoCtrl_getExtRefValueForUnipolar(InstantAoCtrl *_this);
         ErrorCode BufferedAoCtrl_setExtRefValueForUnipolar(InstantAoCtrl *_this, double value);
         double BufferedAoCtrl_getExtRefValueForBipolar(InstantAoCtrl *_this);
         ErrorCode BufferedAoCtrl_setExtRefValueForBipolar(InstantAoCtrl *_this, double value);


         void BufferedAoCtrl_addDataTransmittedListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);
         void BufferedAoCtrl_removeDataTransmittedListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);
         void BufferedAoCtrl_addUnderrunListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);
         void BufferedAoCtrl_removeUnderrunListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);
         void BufferedAoCtrl_addCacheEmptiedListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);
         void BufferedAoCtrl_removeCacheEmptiedListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);
         void BufferedAoCtrl_addTransitStoppedListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);
         void BufferedAoCtrl_removeTransitStoppedListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);
         void BufferedAoCtrl_addStoppedListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);
         void BufferedAoCtrl_removeStoppedListener(BufferedAoCtrl *_this, BfdAoEventListener *listener);

         ErrorCode BufferedAoCtrl_Prepare(BufferedAoCtrl *_this);
         ErrorCode BufferedAoCtrl_RunOnce(BufferedAoCtrl *_this);
         ErrorCode BufferedAoCtrl_Start(BufferedAoCtrl *_this);
         ErrorCode BufferedAoCtrl_Stop(BufferedAoCtrl *_this, int32 action);
         ErrorCode BufferedAoCtrl_SetDataI16(BufferedAoCtrl *_this, int32 count, int16 rawData[]);
         ErrorCode BufferedAoCtrl_SetDataI32(BufferedAoCtrl *_this, int32 count, int32 rawData[]);
         ErrorCode BufferedAoCtrl_SetDataF64(BufferedAoCtrl *_this, int32 count, double scaledData[]);
         void BufferedAoCtrl_Release(BufferedAoCtrl *_this);

         void* BufferedAoCtrl_getBuffer(BufferedAoCtrl *_this);
         int32 BufferedAoCtrl_getBufferCapacity(BufferedAoCtrl *_this);
         ControlState BufferedAoCtrl_getState(BufferedAoCtrl *_this);
         ScanChannel* BufferedAoCtrl_getScanChannel(BufferedAoCtrl *_this);
         ConvertClock* BufferedAoCtrl_getConvertClock(BufferedAoCtrl *_this);
         Trigger* BufferedAoCtrl_getTrigger(BufferedAoCtrl *_this);
         int8 BufferedAoCtrl_getStreaming(BufferedAoCtrl *_this);
         ErrorCode BufferedAoCtrl_setStreaming(BufferedAoCtrl *_this, int8 value);
         Trigger* BufferedAoCtrl_getTrigger1(BufferedAoCtrl *_this);




         int8 DiFeatures_getPortProgrammable(DiFeatures *_this);
         int32 DiFeatures_getPortCount(DiFeatures *_this);
         ICollection* DiFeatures_getPortsType(DiFeatures *_this);
         int8 DiFeatures_getDiSupported(DiFeatures *_this);
         int8 DiFeatures_getDoSupported(DiFeatures *_this);
         int32 DiFeatures_getChannelCountMax(DiFeatures *_this);
         ICollection* DiFeatures_getDataMask(DiFeatures *_this);

         int8 DiFeatures_getNoiseFilterSupported(DiFeatures *_this);
         ICollection* DiFeatures_getNoiseFilterOfChannels(DiFeatures *_this);
         void DiFeatures_getNoiseFilterBlockTimeRange(DiFeatures *_this, MathInterval *value);

         int8 DiFeatures_getDiintSupported(DiFeatures *_this);
         int8 DiFeatures_getDiintGateSupported(DiFeatures *_this);
         int8 DiFeatures_getDiCosintSupported(DiFeatures *_this);
         int8 DiFeatures_getDiPmintSupported(DiFeatures *_this);
         ICollection* DiFeatures_getDiintTriggerEdges(DiFeatures *_this);
         ICollection* DiFeatures_getDiintOfChannels(DiFeatures *_this);
         ICollection* DiFeatures_getDiintGateOfChannels(DiFeatures *_this);
         ICollection* DiFeatures_getDiCosintOfPorts(DiFeatures *_this);
         ICollection* DiFeatures_getDiPmintOfPorts(DiFeatures *_this);
         ICollection* DiFeatures_getSnapEventSources(DiFeatures *_this);

         int8 DiFeatures_getBufferedDiSupported(DiFeatures *_this);
         SamplingMethod DiFeatures_getSamplingMethod(DiFeatures *_this);

         ICollection* DiFeatures_getConvertClockSources(DiFeatures *_this);
         void DiFeatures_getConvertClockRange(DiFeatures *_this, MathInterval *value);

         int8 DiFeatures_getBurstScanSupported(DiFeatures *_this);
         ICollection* DiFeatures_getScanClockSources(DiFeatures *_this);
         void DiFeatures_getScanClockRange(DiFeatures *_this, MathInterval *value);
         int32 DiFeatures_getScanCountMax(DiFeatures *_this);

         int8 DiFeatures_getTriggerSupported(DiFeatures *_this);
         int32 DiFeatures_getTriggerCount(DiFeatures *_this);
         ICollection* DiFeatures_getTriggerSources(DiFeatures *_this);
         ICollection* DiFeatures_getTriggerActions(DiFeatures *_this);
         void DiFeatures_getTriggerDelayRange(DiFeatures *_this, MathInterval *value);





         void InstantDiCtrl_Dispose(InstantDiCtrl *_this);
         void InstantDiCtrl_Cleanup(InstantDiCtrl *_this);
         ErrorCode InstantDiCtrl_UpdateProperties(InstantDiCtrl *_this);
         void InstantDiCtrl_addRemovedListener(InstantDiCtrl *_this, DeviceEventListener * listener);
         void InstantDiCtrl_removeRemovedListener(InstantDiCtrl *_this, DeviceEventListener * listener);
         void InstantDiCtrl_addReconnectedListener(InstantDiCtrl *_this, DeviceEventListener * listener);
         void InstantDiCtrl_removeReconnectedListener(InstantDiCtrl *_this, DeviceEventListener * listener);
         void InstantDiCtrl_addPropertyChangedListener(InstantDiCtrl *_this, DeviceEventListener * listener);
         void InstantDiCtrl_removePropertyChangedListener(InstantDiCtrl *_this, DeviceEventListener * listener);
         void InstantDiCtrl_getSelectedDevice(InstantDiCtrl *_this, DeviceInformation *x);
         ErrorCode InstantDiCtrl_setSelectedDevice(InstantDiCtrl *_this, DeviceInformation const *x);
         int8 InstantDiCtrl_getInitialized(InstantDiCtrl *_this);
         int8 InstantDiCtrl_getCanEditProperty(InstantDiCtrl *_this);
         HANDLE InstantDiCtrl_getDevice(InstantDiCtrl *_this);
         HANDLE InstantDiCtrl_getModule(InstantDiCtrl *_this);
         ICollection* InstantDiCtrl_getSupportedDevices(InstantDiCtrl *_this);
         ICollection* InstantDiCtrl_getSupportedModes(InstantDiCtrl *_this);

         int32 InstantDiCtrl_getPortCount(InstantDiCtrl *_this);
         ICollection* InstantDiCtrl_getPortDirection(InstantDiCtrl *_this);

         DiFeatures* InstantDiCtrl_getFeatures(InstantDiCtrl *_this);
         ICollection* InstantDiCtrl_getNoiseFilter(InstantDiCtrl *_this);


         void InstantDiCtrl_addInterruptListener(InstantDiCtrl *_this, DiSnapEventListener * listener);
         void InstantDiCtrl_removeInterruptListener(InstantDiCtrl *_this, DiSnapEventListener * listener);
         void InstantDiCtrl_addChangeOfStateListener(InstantDiCtrl *_this, DiSnapEventListener * listener);
         void InstantDiCtrl_removeChangeOfStateListener(InstantDiCtrl *_this, DiSnapEventListener * listener);
         void InstantDiCtrl_addPatternMatchListener(InstantDiCtrl *_this, DiSnapEventListener * listener);
         void InstantDiCtrl_removePatternMatchListener(InstantDiCtrl *_this, DiSnapEventListener * listener);

         ErrorCode InstantDiCtrl_ReadAny(InstantDiCtrl *_this, int32 portStart, int32 portCount, uint8 data[]);
   ErrorCode InstantDiCtrl_ReadBit(InstantDiCtrl *_this, int32 port, int32 bit, uint8* data);
         ErrorCode InstantDiCtrl_SnapStart(InstantDiCtrl *_this);
         ErrorCode InstantDiCtrl_SnapStop(InstantDiCtrl *_this);

         ICollection* InstantDiCtrl_getDiintChannels(InstantDiCtrl *_this);
         ICollection* InstantDiCtrl_getDiCosintPorts(InstantDiCtrl *_this);
         ICollection* InstantDiCtrl_getDiPmintPorts(InstantDiCtrl *_this);





         void BufferedDiCtrl_Dispose(BufferedDiCtrl *_this);
         void BufferedDiCtrl_Cleanup(BufferedDiCtrl *_this);
         ErrorCode BufferedDiCtrl_UpdateProperties(BufferedDiCtrl *_this);
         void BufferedDiCtrl_addRemovedListener(BufferedDiCtrl *_this, DeviceEventListener * listener);
         void BufferedDiCtrl_removeRemovedListener(BufferedDiCtrl *_this, DeviceEventListener * listener);
         void BufferedDiCtrl_addReconnectedListener(BufferedDiCtrl *_this, DeviceEventListener * listener);
         void BufferedDiCtrl_removeReconnectedListener(BufferedDiCtrl *_this, DeviceEventListener * listener);
         void BufferedDiCtrl_addPropertyChangedListener(BufferedDiCtrl *_this, DeviceEventListener * listener);
         void BufferedDiCtrl_removePropertyChangedListener(BufferedDiCtrl *_this, DeviceEventListener * listener);
         void BufferedDiCtrl_getSelectedDevice(BufferedDiCtrl *_this, DeviceInformation *x);
         ErrorCode BufferedDiCtrl_setSelectedDevice(BufferedDiCtrl *_this, DeviceInformation const *x);
         int8 BufferedDiCtrl_getInitialized(BufferedDiCtrl *_this);
         int8 BufferedDiCtrl_getCanEditProperty(BufferedDiCtrl *_this);
         HANDLE BufferedDiCtrl_getDevice(BufferedDiCtrl *_this);
         HANDLE BufferedDiCtrl_getModule(BufferedDiCtrl *_this);
         ICollection* BufferedDiCtrl_getSupportedDevices(BufferedDiCtrl *_this);
         ICollection* BufferedDiCtrl_getSupportedModes(BufferedDiCtrl *_this);

         int32 BufferedDiCtrl_getPortCount(BufferedDiCtrl *_this);
         ICollection* BufferedDiCtrl_getPortDirection(BufferedDiCtrl *_this);

         DiFeatures* BufferedDiCtrl_getFeatures(BufferedDiCtrl *_this);
         ICollection* BufferedDiCtrl_getNoiseFilter(BufferedDiCtrl *_this);


         void BufferedDiCtrl_addDataReadyListener(BufferedDiCtrl *_this, BfdDiEventListener *listener);
         void BufferedDiCtrl_removeDataReadyListener(BufferedDiCtrl *_this, BfdDiEventListener *listener);
         void BufferedDiCtrl_addOverrunListener(BufferedDiCtrl *_this, BfdDiEventListener *listener);
         void BufferedDiCtrl_removeOverrunListener(BufferedDiCtrl *_this, BfdDiEventListener *listener);
         void BufferedDiCtrl_addCacheOverflowListener(BufferedDiCtrl *_this, BfdDiEventListener *listener);
         void BufferedDiCtrl_removeCacheOverflowListener(BufferedDiCtrl *_this, BfdDiEventListener *listener);
         void BufferedDiCtrl_addStoppedListener(BufferedDiCtrl *_this, BfdDiEventListener *listener);
         void BufferedDiCtrl_removeStoppedListener(BufferedDiCtrl *_this, BfdDiEventListener *listener);

         ErrorCode BufferedDiCtrl_Prepare(BufferedDiCtrl *_this);
         ErrorCode BufferedDiCtrl_RunOnce(BufferedDiCtrl *_this);
         ErrorCode BufferedDiCtrl_Start(BufferedDiCtrl *_this);
         ErrorCode BufferedDiCtrl_Stop(BufferedDiCtrl *_this);
         ErrorCode BufferedDiCtrl_GetData(BufferedDiCtrl *_this, int32 count, uint8 data[]);
         void BufferedDiCtrl_Release(BufferedDiCtrl *_this);

         void* BufferedDiCtrl_getBuffer(BufferedDiCtrl *_this);
         int32 BufferedDiCtrl_getBufferCapacity(BufferedDiCtrl *_this);
         ControlState BufferedDiCtrl_getState(BufferedDiCtrl *_this);
         ScanPort* BufferedDiCtrl_getScanPort(BufferedDiCtrl *_this);
         ConvertClock* BufferedDiCtrl_getConvertClock(BufferedDiCtrl *_this);
         ScanClock* BufferedDiCtrl_getScanClock(BufferedDiCtrl *_this);
         Trigger* BufferedDiCtrl_getTrigger(BufferedDiCtrl *_this);
         int8 BufferedDiCtrl_getStreaming(BufferedDiCtrl *_this);
         ErrorCode BufferedDiCtrl_setStreaming(BufferedDiCtrl *_this, int8 value);




         int8 DoFeatures_getPortProgrammable(DoFeatures *_this);
         int32 DoFeatures_getPortCount(DoFeatures *_this);
         ICollection* DoFeatures_getPortsType(DoFeatures *_this);
         int8 DoFeatures_getDiSupported(DoFeatures *_this);
         int8 DoFeatures_getDoSupported(DoFeatures *_this);
         int32 DoFeatures_getChannelCountMax(DoFeatures *_this);
         ICollection* DoFeatures_getDataMask(DoFeatures *_this);

         ICollection* DoFeatures_getDoFreezeSignalSources(DoFeatures *_this);

         void DoFeatures_getDoReflectWdtFeedIntervalRange(DoFeatures *_this, MathInterval *value);

         int8 DoFeatures_getBufferedDoSupported(DoFeatures *_this);
         SamplingMethod DoFeatures_getSamplingMethod(DoFeatures *_this);

         ICollection* DoFeatures_getConvertClockSources(DoFeatures *_this);
         void DoFeatures_getConvertClockRange(DoFeatures *_this, MathInterval *value);

         int8 DoFeatures_getBurstScanSupported(DoFeatures *_this);
         ICollection* DoFeatures_getScanClockSources(DoFeatures *_this);
         void DoFeatures_getScanClockRange(DoFeatures *_this, MathInterval *value);
         int32 DoFeatures_getScanCountMax(DoFeatures *_this);

         int8 DoFeatures_getTriggerSupported(DoFeatures *_this);
         int32 DoFeatures_getTriggerCount(DoFeatures *_this);
         ICollection* DoFeatures_getTriggerSources(DoFeatures *_this);
         ICollection* DoFeatures_getTriggerActions(DoFeatures *_this);
         void DoFeatures_getTriggerDelayRange(DoFeatures *_this, MathInterval *value);





         void InstantDoCtrl_Dispose(InstantDoCtrl *_this);
         void InstantDoCtrl_Cleanup(InstantDoCtrl *_this);
         ErrorCode InstantDoCtrl_UpdateProperties(InstantDoCtrl *_this);
         void InstantDoCtrl_addRemovedListener(InstantDoCtrl *_this, DeviceEventListener * listener);
         void InstantDoCtrl_removeRemovedListener(InstantDoCtrl *_this, DeviceEventListener * listener);
         void InstantDoCtrl_addReconnectedListener(InstantDoCtrl *_this, DeviceEventListener * listener);
         void InstantDoCtrl_removeReconnectedListener(InstantDoCtrl *_this, DeviceEventListener * listener);
         void InstantDoCtrl_addPropertyChangedListener(InstantDoCtrl *_this, DeviceEventListener * listener);
         void InstantDoCtrl_removePropertyChangedListener(InstantDoCtrl *_this, DeviceEventListener * listener);
         void InstantDoCtrl_getSelectedDevice(InstantDoCtrl *_this, DeviceInformation *x);
         ErrorCode InstantDoCtrl_setSelectedDevice(InstantDoCtrl *_this, DeviceInformation const *x);
         int8 InstantDoCtrl_getInitialized(InstantDoCtrl *_this);
         int8 InstantDoCtrl_getCanEditProperty(InstantDoCtrl *_this);
         HANDLE InstantDoCtrl_getDevice(InstantDoCtrl *_this);
         HANDLE InstantDoCtrl_getModule(InstantDoCtrl *_this);
         ICollection* InstantDoCtrl_getSupportedDevices(InstantDoCtrl *_this);
         ICollection* InstantDoCtrl_getSupportedModes(InstantDoCtrl *_this);

         int32 InstantDoCtrl_getPortCount(InstantDoCtrl *_this);
         ICollection* InstantDoCtrl_getPortDirection(InstantDoCtrl *_this);

         DoFeatures* InstantDoCtrl_getFeatures(InstantDoCtrl *_this);

         ErrorCode InstantDoCtrl_WriteAny(InstantDoCtrl *_this, int32 portStart, int32 portCount, uint8 data[]);
         ErrorCode InstantDoCtrl_ReadAny(InstantDoCtrl *_this, int32 portStart, int32 portCount, uint8 data[]);
   ErrorCode InstantDoCtrl_WriteBit(InstantDoCtrl *_this, int32 port, int32 bit, uint8 data);
         ErrorCode InstantDoCtrl_ReadBit(InstantDoCtrl *_this, int32 port, int32 bit, uint8* data);





         void BufferedDoCtrl_Dispose(BufferedDoCtrl *_this);
         void BufferedDoCtrl_Cleanup(BufferedDoCtrl *_this);
         ErrorCode BufferedDoCtrl_UpdateProperties(BufferedDoCtrl *_this);
         void BufferedDoCtrl_addRemovedListener(BufferedDoCtrl *_this, DeviceEventListener * listener);
         void BufferedDoCtrl_removeRemovedListener(BufferedDoCtrl *_this, DeviceEventListener * listener);
         void BufferedDoCtrl_addReconnectedListener(BufferedDoCtrl *_this, DeviceEventListener * listener);
         void BufferedDoCtrl_removeReconnectedListener(BufferedDoCtrl *_this, DeviceEventListener * listener);
         void BufferedDoCtrl_addPropertyChangedListener(BufferedDoCtrl *_this, DeviceEventListener * listener);
         void BufferedDoCtrl_removePropertyChangedListener(BufferedDoCtrl *_this, DeviceEventListener * listener);
         void BufferedDoCtrl_getSelectedDevice(BufferedDoCtrl *_this, DeviceInformation *x);
         ErrorCode BufferedDoCtrl_setSelectedDevice(BufferedDoCtrl *_this, DeviceInformation const *x);
         int8 BufferedDoCtrl_getInitialized(BufferedDoCtrl *_this);
         int8 BufferedDoCtrl_getCanEditProperty(BufferedDoCtrl *_this);
         HANDLE BufferedDoCtrl_getDevice(BufferedDoCtrl *_this);
         HANDLE BufferedDoCtrl_getModule(BufferedDoCtrl *_this);
         ICollection* BufferedDoCtrl_getSupportedDevices(BufferedDoCtrl *_this);
         ICollection* BufferedDoCtrl_getSupportedModes(BufferedDoCtrl *_this);

         int32 BufferedDoCtrl_getPortCount(BufferedDoCtrl *_this);
         ICollection* BufferedDoCtrl_getPortDirection(BufferedDoCtrl *_this);

         DoFeatures* BufferedDoCtrl_getFeatures(BufferedDoCtrl *_this);


         void BufferedDoCtrl_addDataTransmittedListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);
         void BufferedDoCtrl_removeDataTransmittedListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);
         void BufferedDoCtrl_addUnderrunListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);
         void BufferedDoCtrl_removeUnderrunListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);
         void BufferedDoCtrl_addCacheEmptiedListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);
         void BufferedDoCtrl_removeCacheEmptiedListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);
         void BufferedDoCtrl_addTransitStoppedListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);
         void BufferedDoCtrl_removeTransitStoppedListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);
         void BufferedDoCtrl_addStoppedListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);
         void BufferedDoCtrl_removeStoppedListener(BufferedDoCtrl *_this, BfdDoEventListener *listener);

         ErrorCode BufferedDoCtrl_Prepare(BufferedDoCtrl *_this);
         ErrorCode BufferedDoCtrl_RunOnce(BufferedDoCtrl *_this);
         ErrorCode BufferedDoCtrl_Start(BufferedDoCtrl *_this);
         ErrorCode BufferedDoCtrl_Stop(BufferedDoCtrl *_this, int32 action);
         ErrorCode BufferedDoCtrl_SetData(BufferedDoCtrl *_this, int32 count, uint8 data[]);
         void BufferedDoCtrl_Release(BufferedDoCtrl *_this);

         void* BufferedDoCtrl_getBuffer(BufferedDoCtrl *_this);
         int32 BufferedDoCtrl_getBufferCapacity(BufferedDoCtrl *_this);
         ControlState BufferedDoCtrl_getState(BufferedDoCtrl *_this);
         ScanPort* BufferedDoCtrl_getScanPort(BufferedDoCtrl *_this);
         ConvertClock* BufferedDoCtrl_getConvertClock(BufferedDoCtrl *_this);
         Trigger* BufferedDoCtrl_getTrigger(BufferedDoCtrl *_this);
         int8 BufferedDoCtrl_getStreaming(BufferedDoCtrl *_this);
         ErrorCode BufferedDoCtrl_setStreaming(BufferedDoCtrl *_this, int8 value);




         void CounterCapabilityIndexer_Dispose(CounterCapabilityIndexer *_this);
         int32 CounterCapabilityIndexer_getCount(CounterCapabilityIndexer *_this);
         ICollection* CounterCapabilityIndexer_getItem(CounterCapabilityIndexer *_this, int32 channel);




         int32 EventCounterFeatures_getChannelCountMax(EventCounterFeatures *_this);
         int32 EventCounterFeatures_getResolution(EventCounterFeatures *_this);
         int32 EventCounterFeatures_getDataSize(EventCounterFeatures *_this);
         CounterCapabilityIndexer* EventCounterFeatures_getCapabilities(EventCounterFeatures *_this);
         int8 EventCounterFeatures_getNoiseFilterSupported(EventCounterFeatures *_this);
         ICollection* EventCounterFeatures_getNoiseFilterOfChannels(EventCounterFeatures *_this);
         void EventCounterFeatures_getNoiseFilterBlockTimeRange(EventCounterFeatures *_this, MathInterval *value);





         void EventCounterCtrl_Dispose(EventCounterCtrl *_this);
         void EventCounterCtrl_Cleanup(EventCounterCtrl *_this);
         ErrorCode EventCounterCtrl_UpdateProperties(EventCounterCtrl *_this);
         void EventCounterCtrl_addRemovedListener(EventCounterCtrl *_this, DeviceEventListener * listener);
         void EventCounterCtrl_removeRemovedListener(EventCounterCtrl *_this, DeviceEventListener * listener);
         void EventCounterCtrl_addReconnectedListener(EventCounterCtrl *_this, DeviceEventListener * listener);
         void EventCounterCtrl_removeReconnectedListener(EventCounterCtrl *_this, DeviceEventListener * listener);
         void EventCounterCtrl_addPropertyChangedListener(EventCounterCtrl *_this, DeviceEventListener * listener);
         void EventCounterCtrl_removePropertyChangedListener(EventCounterCtrl *_this, DeviceEventListener * listener);
         void EventCounterCtrl_getSelectedDevice(EventCounterCtrl *_this, DeviceInformation *x);
         ErrorCode EventCounterCtrl_setSelectedDevice(EventCounterCtrl *_this, DeviceInformation const *x);
         int8 EventCounterCtrl_getInitialized(EventCounterCtrl *_this);
         int8 EventCounterCtrl_getCanEditProperty(EventCounterCtrl *_this);
         HANDLE EventCounterCtrl_getDevice(EventCounterCtrl *_this);
         HANDLE EventCounterCtrl_getModule(EventCounterCtrl *_this);
         ICollection* EventCounterCtrl_getSupportedDevices(EventCounterCtrl *_this);
         ICollection* EventCounterCtrl_getSupportedModes(EventCounterCtrl *_this);

         int32 EventCounterCtrl_getChannel(EventCounterCtrl *_this);
         ErrorCode EventCounterCtrl_setChannel(EventCounterCtrl *_this, int32 ch);
         int8 EventCounterCtrl_getEnabled(EventCounterCtrl *_this);
         ErrorCode EventCounterCtrl_setEnabled(EventCounterCtrl *_this, int8 enabled);
         int8 EventCounterCtrl_getRunning(EventCounterCtrl *_this);

         NoiseFilterChannel* EventCounterCtrl_getNoiseFilter(EventCounterCtrl *_this);

         EventCounterFeatures* EventCounterCtrl_getFeatures(EventCounterCtrl *_this);
         int32 EventCounterCtrl_getValue(EventCounterCtrl *_this);




         int32 FreqMeterFeatures_getChannelCountMax(FreqMeterFeatures *_this);
         int32 FreqMeterFeatures_getResolution(FreqMeterFeatures *_this);
         int32 FreqMeterFeatures_getDataSize(FreqMeterFeatures *_this);
         CounterCapabilityIndexer* FreqMeterFeatures_getCapabilities(FreqMeterFeatures *_this);
         int8 FreqMeterFeatures_getNoiseFilterSupported(FreqMeterFeatures *_this);
         ICollection* FreqMeterFeatures_getNoiseFilterOfChannels(FreqMeterFeatures *_this);
         void FreqMeterFeatures_getNoiseFilterBlockTimeRange(FreqMeterFeatures *_this, MathInterval *value);
         ICollection* FreqMeterFeatures_getFmMethods(FreqMeterFeatures *_this);





         void FreqMeterCtrl_Dispose(FreqMeterCtrl *_this);
         void FreqMeterCtrl_Cleanup(FreqMeterCtrl *_this);
         ErrorCode FreqMeterCtrl_UpdateProperties(FreqMeterCtrl *_this);
         void FreqMeterCtrl_addRemovedListener(FreqMeterCtrl *_this, DeviceEventListener * listener);
         void FreqMeterCtrl_removeRemovedListener(FreqMeterCtrl *_this, DeviceEventListener * listener);
         void FreqMeterCtrl_addReconnectedListener(FreqMeterCtrl *_this, DeviceEventListener * listener);
         void FreqMeterCtrl_removeReconnectedListener(FreqMeterCtrl *_this, DeviceEventListener * listener);
         void FreqMeterCtrl_addPropertyChangedListener(FreqMeterCtrl *_this, DeviceEventListener * listener);
         void FreqMeterCtrl_removePropertyChangedListener(FreqMeterCtrl *_this, DeviceEventListener * listener);
         void FreqMeterCtrl_getSelectedDevice(FreqMeterCtrl *_this, DeviceInformation *x);
         ErrorCode FreqMeterCtrl_setSelectedDevice(FreqMeterCtrl *_this, DeviceInformation const *x);
         int8 FreqMeterCtrl_getInitialized(FreqMeterCtrl *_this);
         int8 FreqMeterCtrl_getCanEditProperty(FreqMeterCtrl *_this);
         HANDLE FreqMeterCtrl_getDevice(FreqMeterCtrl *_this);
         HANDLE FreqMeterCtrl_getModule(FreqMeterCtrl *_this);
         ICollection* FreqMeterCtrl_getSupportedDevices(FreqMeterCtrl *_this);
         ICollection* FreqMeterCtrl_getSupportedModes(FreqMeterCtrl *_this);

         int32 FreqMeterCtrl_getChannel(FreqMeterCtrl *_this);
         ErrorCode FreqMeterCtrl_setChannel(FreqMeterCtrl *_this, int32 ch);
         int8 FreqMeterCtrl_getEnabled(FreqMeterCtrl *_this);
         ErrorCode FreqMeterCtrl_setEnabled(FreqMeterCtrl *_this, int8 enabled);
         int8 FreqMeterCtrl_getRunning(FreqMeterCtrl *_this);

         NoiseFilterChannel* FreqMeterCtrl_getNoiseFilter(FreqMeterCtrl *_this);

         FreqMeterFeatures* FreqMeterCtrl_getFeatures(FreqMeterCtrl *_this);
         double FreqMeterCtrl_getValue(FreqMeterCtrl *_this);
         FreqMeasureMethod FreqMeterCtrl_getMethod(FreqMeterCtrl *_this);
         ErrorCode FreqMeterCtrl_setMethod(FreqMeterCtrl *_this, FreqMeasureMethod value);
         double FreqMeterCtrl_getCollectionPeriod(FreqMeterCtrl *_this);
         ErrorCode FreqMeterCtrl_setCollectionPeriod(FreqMeterCtrl *_this, double value);




         int32 OneShotFeatures_getChannelCountMax(OneShotFeatures *_this);
         int32 OneShotFeatures_getResolution(OneShotFeatures *_this);
         int32 OneShotFeatures_getDataSize(OneShotFeatures *_this);
         CounterCapabilityIndexer* OneShotFeatures_getCapabilities(OneShotFeatures *_this);
         int8 OneShotFeatures_getNoiseFilterSupported(OneShotFeatures *_this);
         ICollection* OneShotFeatures_getNoiseFilterOfChannels(OneShotFeatures *_this);
         void OneShotFeatures_getNoiseFilterBlockTimeRange(OneShotFeatures *_this, MathInterval *value);
         void OneShotFeatures_getDelayCountRange(OneShotFeatures *_this, MathInterval *value);





         void OneShotCtrl_Dispose(OneShotCtrl *_this);
         void OneShotCtrl_Cleanup(OneShotCtrl *_this);
         ErrorCode OneShotCtrl_UpdateProperties(OneShotCtrl *_this);
         void OneShotCtrl_addRemovedListener(OneShotCtrl *_this, DeviceEventListener * listener);
         void OneShotCtrl_removeRemovedListener(OneShotCtrl *_this, DeviceEventListener * listener);
         void OneShotCtrl_addReconnectedListener(OneShotCtrl *_this, DeviceEventListener * listener);
         void OneShotCtrl_removeReconnectedListener(OneShotCtrl *_this, DeviceEventListener * listener);
         void OneShotCtrl_addPropertyChangedListener(OneShotCtrl *_this, DeviceEventListener * listener);
         void OneShotCtrl_removePropertyChangedListener(OneShotCtrl *_this, DeviceEventListener * listener);
         void OneShotCtrl_getSelectedDevice(OneShotCtrl *_this, DeviceInformation *x);
         ErrorCode OneShotCtrl_setSelectedDevice(OneShotCtrl *_this, DeviceInformation const *x);
         int8 OneShotCtrl_getInitialized(OneShotCtrl *_this);
         int8 OneShotCtrl_getCanEditProperty(OneShotCtrl *_this);
         HANDLE OneShotCtrl_getDevice(OneShotCtrl *_this);
         HANDLE OneShotCtrl_getModule(OneShotCtrl *_this);
         ICollection* OneShotCtrl_getSupportedDevices(OneShotCtrl *_this);
         ICollection* OneShotCtrl_getSupportedModes(OneShotCtrl *_this);

         int32 OneShotCtrl_getChannel(OneShotCtrl *_this);
         ErrorCode OneShotCtrl_setChannel(OneShotCtrl *_this, int32 ch);
         int8 OneShotCtrl_getEnabled(OneShotCtrl *_this);
         ErrorCode OneShotCtrl_setEnabled(OneShotCtrl *_this, int8 enabled);
         int8 OneShotCtrl_getRunning(OneShotCtrl *_this);

         NoiseFilterChannel* OneShotCtrl_getNoiseFilter(OneShotCtrl *_this);

         void OneShotCtrl_addOneShotListener(OneShotCtrl *_this, CntrEventListener * listener);
         void OneShotCtrl_removeOneShotListener(OneShotCtrl *_this, CntrEventListener * listener);
         OneShotFeatures* OneShotCtrl_getFeatures(OneShotCtrl *_this);
         int32 OneShotCtrl_getDelayCount(OneShotCtrl *_this);
         ErrorCode OneShotCtrl_setDelayCount(OneShotCtrl *_this, int32 value);




         int32 TimerPulseFeatures_getChannelCountMax(TimerPulseFeatures *_this);
         int32 TimerPulseFeatures_getResolution(TimerPulseFeatures *_this);
         int32 TimerPulseFeatures_getDataSize(TimerPulseFeatures *_this);
         CounterCapabilityIndexer* TimerPulseFeatures_getCapabilities(TimerPulseFeatures *_this);
         int8 TimerPulseFeatures_getNoiseFilterSupported(TimerPulseFeatures *_this);
         ICollection* TimerPulseFeatures_getNoiseFilterOfChannels(TimerPulseFeatures *_this);
         void TimerPulseFeatures_getNoiseFilterBlockTimeRange(TimerPulseFeatures *_this, MathInterval *value);
         void TimerPulseFeatures_getTimerFrequencyRange(TimerPulseFeatures *_this, MathInterval *value);
         int8 TimerPulseFeatures_getTimerEventSupported(TimerPulseFeatures *_this);





         void TimerPulseCtrl_Dispose(TimerPulseCtrl *_this);
         void TimerPulseCtrl_Cleanup(TimerPulseCtrl *_this);
         ErrorCode TimerPulseCtrl_UpdateProperties(TimerPulseCtrl *_this);
         void TimerPulseCtrl_addRemovedListener(TimerPulseCtrl *_this, DeviceEventListener * listener);
         void TimerPulseCtrl_removeRemovedListener(TimerPulseCtrl *_this, DeviceEventListener * listener);
         void TimerPulseCtrl_addReconnectedListener(TimerPulseCtrl *_this, DeviceEventListener * listener);
         void TimerPulseCtrl_removeReconnectedListener(TimerPulseCtrl *_this, DeviceEventListener * listener);
         void TimerPulseCtrl_addPropertyChangedListener(TimerPulseCtrl *_this, DeviceEventListener * listener);
         void TimerPulseCtrl_removePropertyChangedListener(TimerPulseCtrl *_this, DeviceEventListener * listener);
         void TimerPulseCtrl_getSelectedDevice(TimerPulseCtrl *_this, DeviceInformation *x);
         ErrorCode TimerPulseCtrl_setSelectedDevice(TimerPulseCtrl *_this, DeviceInformation const *x);
         int8 TimerPulseCtrl_getInitialized(TimerPulseCtrl *_this);
         int8 TimerPulseCtrl_getCanEditProperty(TimerPulseCtrl *_this);
         HANDLE TimerPulseCtrl_getDevice(TimerPulseCtrl *_this);
         HANDLE TimerPulseCtrl_getModule(TimerPulseCtrl *_this);
         ICollection* TimerPulseCtrl_getSupportedDevices(TimerPulseCtrl *_this);
         ICollection* TimerPulseCtrl_getSupportedModes(TimerPulseCtrl *_this);

         int32 TimerPulseCtrl_getChannel(TimerPulseCtrl *_this);
         ErrorCode TimerPulseCtrl_setChannel(TimerPulseCtrl *_this, int32 ch);
         int8 TimerPulseCtrl_getEnabled(TimerPulseCtrl *_this);
         ErrorCode TimerPulseCtrl_setEnabled(TimerPulseCtrl *_this, int8 enabled);
         int8 TimerPulseCtrl_getRunning(TimerPulseCtrl *_this);

         NoiseFilterChannel* TimerPulseCtrl_getNoiseFilter(TimerPulseCtrl *_this);

         void TimerPulseCtrl_addTimerTickListener(TimerPulseCtrl *_this, CntrEventListener * listener);
         void TimerPulseCtrl_removeTimerTickListener(TimerPulseCtrl *_this, CntrEventListener * listener);
         TimerPulseFeatures* TimerPulseCtrl_getFeatures(TimerPulseCtrl *_this);
         double TimerPulseCtrl_getFrequency(TimerPulseCtrl *_this);
         ErrorCode TimerPulseCtrl_setFrequency(TimerPulseCtrl *_this, double value);




         int32 PwMeterFeatures_getChannelCountMax(PwMeterFeatures *_this);
         int32 PwMeterFeatures_getResolution(PwMeterFeatures *_this);
         int32 PwMeterFeatures_getDataSize(PwMeterFeatures *_this);
         CounterCapabilityIndexer* PwMeterFeatures_getCapabilities(PwMeterFeatures *_this);
         int8 PwMeterFeatures_getNoiseFilterSupported(PwMeterFeatures *_this);
         ICollection* PwMeterFeatures_getNoiseFilterOfChannels(PwMeterFeatures *_this);
         void PwMeterFeatures_getNoiseFilterBlockTimeRange(PwMeterFeatures *_this, MathInterval *value);
         ICollection* PwMeterFeatures_getPwmCascadeGroup(PwMeterFeatures *_this);
         int8 PwMeterFeatures_getOverflowEventSupported(PwMeterFeatures *_this);





         void PwMeterCtrl_Dispose(PwMeterCtrl *_this);
         void PwMeterCtrl_Cleanup(PwMeterCtrl *_this);
         ErrorCode PwMeterCtrl_UpdateProperties(PwMeterCtrl *_this);
         void PwMeterCtrl_addRemovedListener(PwMeterCtrl *_this, DeviceEventListener * listener);
         void PwMeterCtrl_removeRemovedListener(PwMeterCtrl *_this, DeviceEventListener * listener);
         void PwMeterCtrl_addReconnectedListener(PwMeterCtrl *_this, DeviceEventListener * listener);
         void PwMeterCtrl_removeReconnectedListener(PwMeterCtrl *_this, DeviceEventListener * listener);
         void PwMeterCtrl_addPropertyChangedListener(PwMeterCtrl *_this, DeviceEventListener * listener);
         void PwMeterCtrl_removePropertyChangedListener(PwMeterCtrl *_this, DeviceEventListener * listener);
         void PwMeterCtrl_getSelectedDevice(PwMeterCtrl *_this, DeviceInformation *x);
         ErrorCode PwMeterCtrl_setSelectedDevice(PwMeterCtrl *_this, DeviceInformation const *x);
         int8 PwMeterCtrl_getInitialized(PwMeterCtrl *_this);
         int8 PwMeterCtrl_getCanEditProperty(PwMeterCtrl *_this);
         HANDLE PwMeterCtrl_getDevice(PwMeterCtrl *_this);
         HANDLE PwMeterCtrl_getModule(PwMeterCtrl *_this);
         ICollection* PwMeterCtrl_getSupportedDevices(PwMeterCtrl *_this);
         ICollection* PwMeterCtrl_getSupportedModes(PwMeterCtrl *_this);

         int32 PwMeterCtrl_getChannel(PwMeterCtrl *_this);
         ErrorCode PwMeterCtrl_setChannel(PwMeterCtrl *_this, int32 ch);
         int8 PwMeterCtrl_getEnabled(PwMeterCtrl *_this);
         ErrorCode PwMeterCtrl_setEnabled(PwMeterCtrl *_this, int8 enabled);
         int8 PwMeterCtrl_getRunning(PwMeterCtrl *_this);

         NoiseFilterChannel* PwMeterCtrl_getNoiseFilter(PwMeterCtrl *_this);

         void PwMeterCtrl_addOverflowListener(PwMeterCtrl *_this, CntrEventListener * listener);
         void PwMeterCtrl_removeOverflowListener(PwMeterCtrl *_this, CntrEventListener * listener);
         PwMeterFeatures* PwMeterCtrl_getFeatures(PwMeterCtrl *_this);
         void PwMeterCtrl_getValue(PwMeterCtrl *_this, PulseWidth *width);




         int32 PwModulatorFeatures_getChannelCountMax(PwModulatorFeatures *_this);
         int32 PwModulatorFeatures_getResolution(PwModulatorFeatures *_this);
         int32 PwModulatorFeatures_getDataSize(PwModulatorFeatures *_this);
         CounterCapabilityIndexer* PwModulatorFeatures_getCapabilities(PwModulatorFeatures *_this);
         int8 PwModulatorFeatures_getNoiseFilterSupported(PwModulatorFeatures *_this);
         ICollection* PwModulatorFeatures_getNoiseFilterOfChannels(PwModulatorFeatures *_this);
         void PwModulatorFeatures_getNoiseFilterBlockTimeRange(PwModulatorFeatures *_this, MathInterval *value);
         void PwModulatorFeatures_getHiPeriodRange(PwModulatorFeatures *_this, MathInterval *value);
         void PwModulatorFeatures_getLoPeriodRange(PwModulatorFeatures *_this, MathInterval *value);





         void PwModulatorCtrl_Dispose(PwModulatorCtrl *_this);
         void PwModulatorCtrl_Cleanup(PwModulatorCtrl *_this);
         ErrorCode PwModulatorCtrl_UpdateProperties(PwModulatorCtrl *_this);
         void PwModulatorCtrl_addRemovedListener(PwModulatorCtrl *_this, DeviceEventListener * listener);
         void PwModulatorCtrl_removeRemovedListener(PwModulatorCtrl *_this, DeviceEventListener * listener);
         void PwModulatorCtrl_addReconnectedListener(PwModulatorCtrl *_this, DeviceEventListener * listener);
         void PwModulatorCtrl_removeReconnectedListener(PwModulatorCtrl *_this, DeviceEventListener * listener);
         void PwModulatorCtrl_addPropertyChangedListener(PwModulatorCtrl *_this, DeviceEventListener * listener);
         void PwModulatorCtrl_removePropertyChangedListener(PwModulatorCtrl *_this, DeviceEventListener * listener);
         void PwModulatorCtrl_getSelectedDevice(PwModulatorCtrl *_this, DeviceInformation *x);
         ErrorCode PwModulatorCtrl_setSelectedDevice(PwModulatorCtrl *_this, DeviceInformation const *x);
         int8 PwModulatorCtrl_getInitialized(PwModulatorCtrl *_this);
         int8 PwModulatorCtrl_getCanEditProperty(PwModulatorCtrl *_this);
         HANDLE PwModulatorCtrl_getDevice(PwModulatorCtrl *_this);
         HANDLE PwModulatorCtrl_getModule(PwModulatorCtrl *_this);
         ICollection* PwModulatorCtrl_getSupportedDevices(PwModulatorCtrl *_this);
         ICollection* PwModulatorCtrl_getSupportedModes(PwModulatorCtrl *_this);

         int32 PwModulatorCtrl_getChannel(PwModulatorCtrl *_this);
         ErrorCode PwModulatorCtrl_setChannel(PwModulatorCtrl *_this, int32 ch);
         int8 PwModulatorCtrl_getEnabled(PwModulatorCtrl *_this);
         ErrorCode PwModulatorCtrl_setEnabled(PwModulatorCtrl *_this, int8 enabled);
         int8 PwModulatorCtrl_getRunning(PwModulatorCtrl *_this);

         NoiseFilterChannel* PwModulatorCtrl_getNoiseFilter(PwModulatorCtrl *_this);

         PwModulatorFeatures* PwModulatorCtrl_getFeatures(PwModulatorCtrl *_this);
         void PwModulatorCtrl_getPulseWidth(PwModulatorCtrl *_this, PulseWidth *width);
         ErrorCode PwModulatorCtrl_setPulseWidth(PwModulatorCtrl *_this, PulseWidth *width);




         int32 UdCounterFeatures_getChannelCountMax(UdCounterFeatures *_this);
         int32 UdCounterFeatures_getResolution(UdCounterFeatures *_this);
         int32 UdCounterFeatures_getDataSize(UdCounterFeatures *_this);
         CounterCapabilityIndexer* UdCounterFeatures_getCapabilities(UdCounterFeatures *_this);
         int8 UdCounterFeatures_getNoiseFilterSupported(UdCounterFeatures *_this);
         ICollection* UdCounterFeatures_getNoiseFilterOfChannels(UdCounterFeatures *_this);
         void UdCounterFeatures_getNoiseFilterBlockTimeRange(UdCounterFeatures *_this, MathInterval *value);
         ICollection* UdCounterFeatures_getCountingTypes(UdCounterFeatures *_this);
         ICollection* UdCounterFeatures_getInitialValues(UdCounterFeatures *_this);
         ICollection* UdCounterFeatures_getSnapEventSources(UdCounterFeatures *_this);





         void UdCounterCtrl_Dispose(UdCounterCtrl *_this);
         void UdCounterCtrl_Cleanup(UdCounterCtrl *_this);
         ErrorCode UdCounterCtrl_UpdateProperties(UdCounterCtrl *_this);
         void UdCounterCtrl_addRemovedListener(UdCounterCtrl *_this, DeviceEventListener * listener);
         void UdCounterCtrl_removeRemovedListener(UdCounterCtrl *_this, DeviceEventListener * listener);
         void UdCounterCtrl_addReconnectedListener(UdCounterCtrl *_this, DeviceEventListener * listener);
         void UdCounterCtrl_removeReconnectedListener(UdCounterCtrl *_this, DeviceEventListener * listener);
         void UdCounterCtrl_addPropertyChangedListener(UdCounterCtrl *_this, DeviceEventListener * listener);
         void UdCounterCtrl_removePropertyChangedListener(UdCounterCtrl *_this, DeviceEventListener * listener);
         void UdCounterCtrl_getSelectedDevice(UdCounterCtrl *_this, DeviceInformation *x);
         ErrorCode UdCounterCtrl_setSelectedDevice(UdCounterCtrl *_this, DeviceInformation const *x);
         int8 UdCounterCtrl_getInitialized(UdCounterCtrl *_this);
         int8 UdCounterCtrl_getCanEditProperty(UdCounterCtrl *_this);
         HANDLE UdCounterCtrl_getDevice(UdCounterCtrl *_this);
         HANDLE UdCounterCtrl_getModule(UdCounterCtrl *_this);
         ICollection* UdCounterCtrl_getSupportedDevices(UdCounterCtrl *_this);
         ICollection* UdCounterCtrl_getSupportedModes(UdCounterCtrl *_this);

         int32 UdCounterCtrl_getChannel(UdCounterCtrl *_this);
         ErrorCode UdCounterCtrl_setChannel(UdCounterCtrl *_this, int32 ch);
         int8 UdCounterCtrl_getEnabled(UdCounterCtrl *_this);
         ErrorCode UdCounterCtrl_setEnabled(UdCounterCtrl *_this, int8 enabled);
         int8 UdCounterCtrl_getRunning(UdCounterCtrl *_this);

         NoiseFilterChannel* UdCounterCtrl_getNoiseFilter(UdCounterCtrl *_this);

         void UdCounterCtrl_addUdCntrEventListener(UdCounterCtrl *_this, UdCntrEventListener * listener);
         void UdCounterCtrl_removeUdCntrEventListener(UdCounterCtrl *_this, UdCntrEventListener * listener);
         ErrorCode UdCounterCtrl_SnapStart(UdCounterCtrl *_this, int32 srcId);
         ErrorCode UdCounterCtrl_SnapStop(UdCounterCtrl *_this, int32 srcId);
         ErrorCode UdCounterCtrl_CompareSetTable(UdCounterCtrl *_this, int32 count, int32 *table);
         ErrorCode UdCounterCtrl_CompareSetInterval(UdCounterCtrl *_this, int32 start, int32 increment,int32 count);
         ErrorCode UdCounterCtrl_CompareClear(UdCounterCtrl *_this);
         ErrorCode UdCounterCtrl_ValueReset(UdCounterCtrl *_this);

         UdCounterFeatures* UdCounterCtrl_getFeatures(UdCounterCtrl *_this);
         int32 UdCounterCtrl_getValue(UdCounterCtrl *_this);
         SignalCountingType UdCounterCtrl_getCountingType(UdCounterCtrl *_this);
         ErrorCode UdCounterCtrl_setCountingType(UdCounterCtrl *_this, SignalCountingType value);
         int32 UdCounterCtrl_getInitialValue(UdCounterCtrl *_this);
         ErrorCode UdCounterCtrl_setInitialValue(UdCounterCtrl *_this, int32 value);
         int32 UdCounterCtrl_getResetTimesByIndex(UdCounterCtrl *_this);
         ErrorCode UdCounterCtrl_setResetTimesByIndex(UdCounterCtrl *_this, int32 value);
# 5046 "bdaqctrl.h"

