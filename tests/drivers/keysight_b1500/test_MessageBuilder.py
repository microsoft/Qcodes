import re

import pytest

import qcodes.instrument_drivers.Keysight.keysightb1500.constants as c
from qcodes.instrument_drivers.Keysight.keysightb1500.message_builder import (
    MessageBuilder,
    as_csv,
)


@pytest.fixture
def mb():
    yield MessageBuilder()


def test_as_csv() -> None:
    assert "1" == as_csv([1])

    assert "1,2,3" == as_csv([1, 2, 3])

    assert "1,2" == as_csv([c.ChNr.SLOT_01_CH1, c.ChNr.SLOT_02_CH1])


def _skip():
    pytest.skip("not implemented yet")


def test_cmd(mb) -> None:
    cmd = mb.aad(1, 0).ach(1, 5).ab()
    assert "AAD 1,0;ACH 1,5;AB" == cmd.message


def test_raise_error_on_appending_command_after_final_command(mb) -> None:
    with pytest.raises(ValueError):
        mb.aad(1, 0).ab().ach(1, 5)


def test_exception_on_too_long_message(mb) -> None:
    length = 0
    while length < 250:
        mb.os()
        length += 3  # "OS;" <- three characters per iteration

    match = re.escape(
        f"Command is too long ({len(str(mb._msg))}>256-termchars) "
        f"and will overflow input buffer of instrument. "
        f"(Consider using the ST/END/DO/RU commands for very long "
        f"programs.)"
    )
    with pytest.raises(Exception, match=match):
        _ = mb.message


def test_clear_message_queue(mb) -> None:
    mb.aad(1, 0)
    assert mb.message == "AAD 1,0"

    mb.clear_message_queue()

    assert mb.message == ""


def test_aad(mb) -> None:
    assert "AAD 1,0" == mb.aad(c.ChNr.SLOT_01_CH1, c.AAD.Type.HIGH_SPEED).message

    mb.clear_message_queue()

    assert (
        "AAD 1,1"
        == mb.aad(c.ChNr.SLOT_01_CH1, adc_type=c.AAD.Type.HIGH_RESOLUTION).message
    )


def test_ab(mb) -> None:
    assert "AB" == mb.ab().message


def test_ach(mb) -> None:
    assert "ACH 1,5" == mb.ach(c.ChNr.SLOT_01_CH1, c.ChNr.SLOT_05_CH1).message

    mb.clear_message_queue()

    assert "ACH 1" == mb.ach(c.ChNr.SLOT_01_CH1).message

    mb.clear_message_queue()

    assert "ACH" == mb.ach().message


def test_act(mb) -> None:
    assert "ACT 0,1" == mb.act(c.ACT.Mode.AUTO, 1).message

    mb.clear_message_queue()

    assert "ACT 2" == mb.act(c.ACT.Mode.PLC).message


def test_acv(mb) -> None:
    assert "ACV 7,0.01" == mb.acv(c.ChNr.SLOT_07_CH1, 0.01).message


def test_adj(mb) -> None:
    assert "ADJ 9,1" == mb.adj(c.ChNr.SLOT_09_CH1, c.ADJ.Mode.MANUAL).message


def test_adj_query(mb) -> None:
    assert "ADJ? 1" == mb.adj_query(c.ChNr.SLOT_01_CH1).message

    mb.clear_message_queue()

    assert (
        "ADJ? 1,1" == mb.adj_query(c.ChNr.SLOT_01_CH1, c.ADJQuery.Mode.MEASURE).message
    )


def test_ait(mb) -> None:
    assert (
        "AIT 2,3,0.001"
        == mb.ait(
            c.AIT.Type.HIGH_SPEED_PULSED, c.AIT.Mode.MEAS_TIME_MODE, 0.001
        ).message
    )
    mb.clear_message_queue()
    assert (
        "AIT 2,3"
        == mb.ait(c.AIT.Type.HIGH_SPEED_PULSED, c.AIT.Mode.MEAS_TIME_MODE).message
    )


def test_aitm(mb) -> None:
    assert "AITM 0" == mb.aitm(c.APIVersion.B1500).message


def test_aitm_query(mb: MessageBuilder) -> None:
    assert "AITM?" == mb.aitm_query().message


def test_als(mb) -> None:
    with pytest.raises(NotImplementedError):
        mb.als(c.ChNr.SLOT_01_CH1, 1, b"a")


def test_als_query(mb) -> None:
    assert "ALS? 1" == mb.als_query(c.ChNr.SLOT_01_CH1).message


def test_alw(mb) -> None:
    with pytest.raises(NotImplementedError):
        mb.alw(c.ChNr.SLOT_01_CH1, 1, b"a")


def test_alw_query(mb) -> None:
    assert "ALW? 1" == mb.alw_query(c.ChNr.SLOT_01_CH1).message


def test_av(mb) -> None:
    assert "AV 10" == mb.av(10).message
    mb.clear_message_queue()
    assert "AV -50" == mb.av(-50).message
    mb.clear_message_queue()
    assert "AV 100,1" == mb.av(100, c.AV.Mode.MANUAL).message


def test_az(mb) -> None:
    assert "AZ 0" == mb.az(False).message
    mb.clear_message_queue()
    assert "AZ 1" == mb.az(True).message


def test_bc(mb) -> None:
    assert "BC" == mb.bc().message


def test_bdm(mb: MessageBuilder) -> None:
    assert "BDM 0,1" == mb.bdm(c.BDM.Interval.SHORT, c.BDM.Mode.CURRENT).message


def test_bdt(mb: MessageBuilder) -> None:
    assert "BDT 0.1,0.001" == mb.bdt(hold=0.1, delay=1e-3).message


def test_bdv(mb) -> None:
    assert (
        "BDV 1,0,0,100,0.01"
        == mb.bdv(
            chnum=c.ChNr.SLOT_01_CH1,
            v_range=c.VOutputRange.AUTO,
            start=0,
            stop=100,
            i_comp=0.01,
        ).message
    )


def test_bgi(mb) -> None:
    assert (
        "BGI 1,0,1e-08,14,1e-06"
        == mb.bgi(
            chnum=c.ChNr.SLOT_01_CH1,
            searchmode=c.BinarySearchMode.LIMIT,
            stop_condition=1e-8,
            i_range=14,
            target=1e-6,
        ).message
    )


def test_bgv(mb) -> None:
    assert "BGV 1,0,0.1,12,5" == mb.bgv(1, 0, 0.1, 12, 5).message


def test_bsi(mb) -> None:
    assert "BSI 1,0,1e-12,1e-06,10" == mb.bsi(1, 0, 1e-12, 1e-6, 10).message


def test_bsm(mb) -> None:
    assert "BSM 1,2,3" == mb.bsm(1, 2, 3).message


def test_bssi(mb) -> None:
    assert "BSSI 1,0,1e-06,10" == mb.bssi(1, 0, 1e-6, 10).message


def test_bssv(mb) -> None:
    assert "BSSV 1,0,5,1e-06" == mb.bssv(1, 0, 5, 1e-6).message


def test_bst(mb) -> None:
    assert "BST 5,0.1" == mb.bst(5, 0.1).message


def test_bsv(mb) -> None:
    assert "BSV 1,0,0,20,1e-06" == mb.bsv(1, 0, 0, 20, 1e-6).message


def test_bsvm(mb) -> None:
    assert "BSVM 1" == mb.bsvm(1).message


def test_ca(mb) -> None:
    assert "CA" == mb.ca().message


def test_cal_query(mb) -> None:
    assert "*CAL?" == mb.cal_query().message


def test_cl(mb) -> None:
    assert "CL" == mb.cl().message
    mb.clear_message_queue()
    assert "CL 1,2,3,5" == mb.cl([1, 2, 3, 5]).message


def test_clcorr(mb) -> None:
    assert "CLCORR 9,1" == mb.clcorr(9, 1).message


def test_cm(mb) -> None:
    assert "CM 0" == mb.cm(False).message


def test_cmm(mb) -> None:
    assert "CMM 1,2" == mb.cmm(1, 2).message


def test_cn(mb) -> None:
    assert "CN" == mb.cn().message
    mb.clear_message_queue()
    assert "CN 1,2,3,5" == mb.cn([1, 2, 3, 5]).message


def test_cnx(mb) -> None:
    assert "CNX" == mb.cnx().message
    mb.clear_message_queue()
    assert "CNX 1,2,3,5" == mb.cnx([1, 2, 3, 5]).message


def test_corr_query(mb) -> None:
    assert "CORR? 9,3" == mb.corr_query(9, 3).message


def test_corrdt(mb) -> None:
    assert (
        "CORRDT 9,3000000,0,0,0,0,0,0"
        == mb.corrdt(9, 3000000, 0, 0, 0, 0, 0, 0).message
    )


def test_corrdt_query(mb) -> None:
    assert "CORRDT? 9,1" == mb.corrdt_query(9, 1).message


def test_corrl(mb) -> None:
    assert "CORRL 9,3000000" == mb.corrl(9, 3000000).message


def test_corrl_query(mb) -> None:
    assert "CORRL? 9" == mb.corrl_query(9).message
    mb.clear_message_queue()
    assert "CORRL? 9,2" == mb.corrl_query(9, 2).message


def test_corrser_query(mb: MessageBuilder) -> None:
    assert (
        "CORRSER? 101,1,1e-07,1e-08,10"
        == mb.corrser_query(101, True, 1e-7, 1e-8, 10).message
    )


def test_corrst(mb) -> None:
    assert "CORRST 3,1,1" == mb.corrst(3, 1, 1).message


def test_corrst_query(mb) -> None:
    assert "CORRST? 1,2" == mb.corrst_query(1, 2).message


def test_dcorr(mb) -> None:
    assert "DCORR 3,1,100,0.1,0.2" == mb.dcorr(3, 1, 100, 0.1, 0.2).message


def test_dcorr_query(mb) -> None:
    assert "DCORR? 3,1" == mb.dcorr_query(3, 1).message


def test_dcv(mb) -> None:
    assert "DCV 3,2.5" == mb.dcv(3, 2.5).message


def test_di(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_diag_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_do(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_dsmplarm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_dsmplflush(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_dsmplsetup(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_dv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_dz(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_emg_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_end(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmaa(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmaa_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmagrd(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmagrd_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmaio(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmaio_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmapfgd(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpa(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpa_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpe(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpe_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpl(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpl_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpp_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpqg(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpqg_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpr(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpr_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhps(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhps_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvca(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvca_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvctst_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvp_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvpv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvs(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvs_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ermod(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ermod_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfda(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfda_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfdp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfdp_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfds(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfds_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfga(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfga_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfgp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfgp_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfgr(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfgr_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfqg(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfqg_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpftemp_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhca(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhca_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhccal_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhcmax_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhctst(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_err_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_errx_query(mb) -> None:
    assert "ERRX?" == mb.errx_query().message
    mb.clear_message_queue()
    assert "ERRX? 0" == mb.errx_query(mode=c.ERRX.Mode.CODE_AND_MESSAGE).message


def test_ers_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erssp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_erssp_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_eruhva(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_eruhva_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_fc(mb) -> None:
    assert "FC 3,1000" == mb.fc(3, 1000).message


def test_fl(mb) -> None:
    assert "FL 1" == mb.fl(True).message
    mb.clear_message_queue()
    assert "FL 0,1,3,5" == mb.fl(False, [1, 3, 5]).message
    mb.clear_message_queue()
    assert (
        "FL 0,1,2,3,4,5,6,7,8,9,10"
        == mb.fl(False, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).message
    )
    mb.clear_message_queue()
    with pytest.raises(ValueError):
        _ = mb.fl(False, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).message


def test_fmt(mb) -> None:
    assert "FMT 1" == mb.fmt(c.FMT.Format.ASCII_12_DIGITS_WITH_HEADER_CRLF_EOI).message
    mb.clear_message_queue()
    assert (
        "FMT 2,1"
        == mb.fmt(
            c.FMT.Format.ASCII_12_DIGITS_NO_HEADER_CRLF_EOI,
            c.FMT.Mode.PRIMARY_SOURCE_OUTPUT_DATA,
        ).message
    )


def test_hvsmuop(mb) -> None:
    _skip()


def test_hvsmuop_query(mb) -> None:
    assert "HVSMUOP?" == mb.hvsmuop_query().message


def test_idn_query(mb) -> None:
    assert "IN" == mb.in_().message
    mb.clear_message_queue()
    assert "IN 1,2,3,5,6" == mb.in_([1, 2, 3, 5, 6]).message


def test_imp(mb) -> None:
    assert "IMP 10" == mb.imp(c.IMP.MeasurementMode.Z_THETA_RAD).message


def test_in_(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_intlkvth(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_intlkvth_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lgi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lgv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lim(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lim_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lmn(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lop_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lrn_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lsi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lsm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lssi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lssv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lst_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lstm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lsv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_lsvm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpnt(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpnx(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpt(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpws(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpwnx(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ml(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mm(mb) -> None:
    assert "MM 1,1" == mb.mm(mode=c.MM.Mode.SPOT, channels=[c.ChNr.SLOT_01_CH1]).message
    mb.clear_message_queue()
    assert (
        "MM 2,1,3"
        == mb.mm(
            mode=c.MM.Mode.STAIRCASE_SWEEP,
            channels=[c.ChNr.SLOT_01_CH1, c.ChNr.SLOT_03_CH1],
        ).message
    )


def test_msc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_msp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mt(mb) -> None:
    assert "MT 0.0,0.42,32" == mb.mt(0.0, 0.42, 32).message
    mb.clear_message_queue()
    assert "MT 0.0,0.42,32,0.12" == mb.mt(0.0, 0.42, 32, 0.12).message


def test_mtdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_mv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_nub_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_odsw(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_odsw_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_opc_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_os(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_osx(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pa(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pad(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pax(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pch(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pch_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pt(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ptdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pwdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pwi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_pwv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsl(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_qso(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsr(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_qst(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsz(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_rc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_rcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ri(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_rm(mb) -> None:
    assert "RM 1,2" == mb.rm(1, c.RM.Mode.AUTO_UP).message
    mb.clear_message_queue()
    assert "RM 2,3,60" == mb.rm(2, c.RM.Mode.AUTO_UP_DOWN, 60).message
    mb.clear_message_queue()
    with pytest.raises(ValueError):
        mb.rm(c.ChNr.SLOT_01_CH1, c.RM.Mode.DEFAULT, 22)


def test_rst(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ru(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_rv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_rz(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sal(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sap(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sar(mb) -> None:
    assert "SAR 1,0" == mb.sar(1, enable_picoamp_autoranging=True).message


def test_scr(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ser(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ser_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sim(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sim_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sopc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sopc_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sovc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sovc_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spm_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spper(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spper_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sprm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sprm_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spst_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spt(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spt_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spupd(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_spv_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sre(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_sre_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_srp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ssl(mb) -> None:
    assert "SSL 9,0" == mb.ssl(9, enable_indicator_led=False).message


def test_ssp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ssr(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_st(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_stb_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_stgp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_stgp_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tacv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tc(mb) -> None:
    assert "TC 3,0" == mb.tc(3, 0).message


def test_tdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tdi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tdv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgmo(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgp(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgpc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgsi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgso(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgxo(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ti(mb) -> None:
    assert "TI 1" == mb.ti(chnum=1).message
    mb.clear_message_queue()
    assert "TI 1,-14" == mb.ti(chnum=1, i_range=c.IMeasRange.FIX_1uA).message


def test_tiv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tmacv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tmdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tsc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tsq(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tsr(mb) -> None:
    assert "TSR" == mb.tsr().message
    mb.clear_message_queue()
    assert "TSR 1" == mb.tsr(c.ChNr.SLOT_01_CH1).message


def test_tst(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ttc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tti(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ttiv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ttv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_tv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_unt_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_var(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_var_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wacv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wat(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wfc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wm(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wmacv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wmdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wmfc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wncc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wnu_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wnx(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_ws(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wsi(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wsv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wt(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wtacv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wtdcv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wtfc(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wv(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_wz_query(mb) -> None:
    # assert '' == mb.().message
    _skip()


def test_xe(mb) -> None:
    assert "XE" == mb.xe().message


def test_nplc_setting_for_high_speed_vs_high_resolution_mode(mb) -> None:
    msg = (
        mb.ait(adc_type=c.AIT.Type.HIGH_SPEED, mode=c.AIT.Mode.NPLC, coeff=3)
        .ait(adc_type=c.AIT.Type.HIGH_RESOLUTION, mode=c.AIT.Mode.NPLC, coeff=8)
        .message
    )
    assert msg == "AIT 0,2,3;AIT 1,2,8"


def test_set_resolution_mode_for_each_smu(mb) -> None:
    msg = (
        mb.aad(chnum=1, adc_type=c.AAD.Type.HIGH_SPEED)
        .aad(chnum=2, adc_type=c.AAD.Type.HIGH_RESOLUTION)
        .message
    )
    assert msg == "AAD 1,0;AAD 2,1"
