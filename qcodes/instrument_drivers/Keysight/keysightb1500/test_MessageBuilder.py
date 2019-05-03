from qcodes.instrument_drivers.Keysight.keysightb1500 import \
    MessageBuilder

import qcodes.instrument_drivers.Keysight.keysightb1500.constants as c

import pytest


@pytest.fixture
def mb() -> MessageBuilder:
    yield MessageBuilder()


def test_as_csv():
    from qcodes.instrument_drivers.Keysight.keysightb1500.message_builder \
        import as_csv

    assert '1' == as_csv([1])

    assert '1,2,3' == as_csv([1, 2, 3])

    assert '1,2' == as_csv([c.ChNr.SLOT_01_CH1, c.ChNr.SLOT_02_CH1])


class TestMessageBuilder:
    def skip(self):
        pytest.skip("not implemented yet")

    def test_cmd(self, mb):
        c = mb.aad(1, 0).ach(1, 5).ab()
        assert 'AAD 1,0;ACH 1,5;AB' == c.message

    def test_raise_error_on_appending_command_after_final_command(self, mb):
        with pytest.raises(ValueError):
            c = mb.aad(1, 0).ab().ach(1, 5)

    def test_warning_on_too_long_message(self, mb):
        length = 0
        while length < 250:
            mb.os()
            length += 3  # "OS;" <- three characters per iteration

        with pytest.warns(UserWarning):
            assert len(mb.message) > 250

    def test_clear_message_queue(self, mb):
        mb.aad(1, 0)
        assert mb.message == 'AAD 1,0'

        mb.clear_message_queue()

        assert mb.message == ''

    def test_aad(self, mb):
        assert 'AAD 1,0' == \
               mb.aad(c.ChNr.SLOT_01_CH1,
                      c.AAD.Type.HIGH_SPEED).message

        mb.clear_message_queue()

        assert 'AAD 1,1' == \
               mb.aad(c.ChNr.SLOT_01_CH1,
                      adc_type=c.AAD.Type.HIGH_RESOLUTION).message

    def test_ab(self, mb):
        assert 'AB' == mb.ab().message

    def test_ach(self, mb):
        assert 'ACH 1,5' == mb.ach(c.ChNr.SLOT_01_CH1,
                                   c.ChNr.SLOT_05_CH1).message

        mb.clear_message_queue()

        assert 'ACH 1' == mb.ach(c.ChNr.SLOT_01_CH1).message

        mb.clear_message_queue()

        assert 'ACH' == mb.ach().message

    def test_act(self, mb):
        assert 'ACT 0,1' == mb.act(c.ACT.Mode.AUTO, 1).message

        mb.clear_message_queue()

        assert 'ACT 2' == mb.act(c.ACT.Mode.PLC).message

    def test_acv(self, mb):
        assert 'ACV 7,0.01' == \
               mb.acv(c.ChNr.SLOT_07_CH1, 0.01).message

    def test_adj(self, mb):
        assert 'ADJ 9,1' == mb.adj(c.ChNr.SLOT_09_CH1,
                                   c.ADJ.Mode.MANUAL).message

    def test_adj_query(self, mb):
        assert 'ADJ? 1' == mb.adj_query(c.ChNr.SLOT_01_CH1).message

        mb.clear_message_queue()

        assert 'ADJ? 1,1' == mb.adj_query(c.ChNr.SLOT_01_CH1,
                                          c.ADJQuery.Mode.MEASURE).message

    def test_ait(self, mb):
        assert 'AIT 2,3,0.001' == mb.ait(c.AIT.Type.HIGH_SPEED_PULSED,
                                         c.AIT.Mode.MEAS_TIME_MODE,
                                         0.001).message
        mb.clear_message_queue()
        assert 'AIT 2,3' == mb.ait(c.AIT.Type.HIGH_SPEED_PULSED,
                                   c.AIT.Mode.MEAS_TIME_MODE).message

    def test_aitm(self, mb):
        assert 'AITM 0' == mb.aitm(c.APIVersion.B1500).message

    def test_aitm_query(self, mb: MessageBuilder):
        assert 'AITM?' == mb.aitm_query().message

    def test_als(self, mb):
        with pytest.raises(NotImplementedError):
            mb.als(c.ChNr.SLOT_01_CH1, 1, b'a')

    def test_als_query(self, mb):
        assert 'ALS? 1' == mb.als_query(c.ChNr.SLOT_01_CH1).message

    def test_alw(self, mb):
        with pytest.raises(NotImplementedError):
            mb.alw(c.ChNr.SLOT_01_CH1, 1, b'a')

    def test_alw_query(self, mb):
        assert 'ALW? 1' == mb.alw_query(c.ChNr.SLOT_01_CH1).message

    def test_av(self, mb):
        assert 'AV 10' == mb.av(10).message
        mb.clear_message_queue()
        assert 'AV -50' == mb.av(-50).message
        mb.clear_message_queue()
        assert 'AV 100,1' == mb.av(100, c.AV.Mode.MANUAL).message

    def test_az(self, mb):
        assert 'AZ 0' == mb.az(False).message

    def test_bc(self, mb):
        assert 'BC' == mb.bc().message

    def test_bdm(self, mb: MessageBuilder):
        assert 'BDM 0,1' == mb.bdm(c.BDM.Interval.SHORT,
                                   c.BDM.Mode.CURRENT).message

    def test_bdt(self, mb: MessageBuilder):
        assert 'BDT 0.1,0.001' == mb.bdt(hold=0.1, delay=1e-3).message

    def test_bdv(self, mb):
        assert 'BDV 1,0,0,100,0.01' == mb.bdv(chnum=c.ChNr.SLOT_01_CH1,
                                              v_range=c.VOutputRange.AUTO,
                                              start=0, stop=100,
                                              i_comp=0.01).message

    def test_bgi(self, mb):
        assert 'BGI 1,0,1e-08,14,1e-06' == \
               mb.bgi(chnum=c.ChNr.SLOT_01_CH1,
                      searchmode=c.BinarySearchMode.LIMIT,
                      stop_condition=1e-8,
                      i_range=14,
                      target=1e-6).message

    def test_bgv(self, mb):
        assert 'BGV 1,0,0.1,12,5' == mb.bgv(1, 0, 0.1, 12, 5).message

    def test_bsi(self, mb):
        assert 'BSI 1,0,1e-12,1e-06,10' == mb.bsi(1, 0, 1e-12, 1e-6,
                                                  10).message

    def test_bsm(self, mb):
        assert 'BSM 1,2,3' == mb.bsm(1, 2, 3).message

    def test_bssi(self, mb):
        assert 'BSSI 1,0,1e-06,10' == mb.bssi(1, 0, 1e-6, 10).message

    def test_bssv(self, mb):
        assert 'BSSV 1,0,5,1e-06' == mb.bssv(1, 0, 5, 1e-6).message

    def test_bst(self, mb):
        assert 'BST 5,0.1' == mb.bst(5, 0.1).message

    def test_bsv(self, mb):
        assert 'BSV 1,0,0,20,1e-06' == mb.bsv(1, 0, 0, 20, 1e-6).message

    def test_bsvm(self, mb):
        assert 'BSVM 1' == mb.bsvm(1).message

    def test_ca(self, mb):
        assert 'CA' == mb.ca().message

    def test_cal_query(self, mb):
        assert '*CAL?' == mb.cal_query().message

    def test_cl(self, mb):
        assert 'CL' == mb.cl().message
        mb.clear_message_queue()
        assert 'CL 1,2,3,5' == mb.cl([1, 2, 3, 5]).message

    def test_clcorr(self, mb):
        assert 'CLCORR 9,1' == mb.clcorr(9, 1).message

    def test_cm(self, mb):
        assert 'CM 0' == mb.cm(False).message

    def test_cmm(self, mb):
        assert 'CMM 1,2' == mb.cmm(1, 2).message

    def test_cn(self, mb):
        assert 'CN' == mb.cn().message
        mb.clear_message_queue()
        assert 'CN 1,2,3,5' == mb.cn([1, 2, 3, 5]).message

    def test_cnx(self, mb):
        assert 'CNX' == mb.cnx().message
        mb.clear_message_queue()
        assert 'CNX 1,2,3,5' == mb.cnx([1, 2, 3, 5]).message

    def test_corr_query(self, mb):
        assert 'CORR? 9,3' == mb.corr_query(9, 3).message

    def test_corrdt(self, mb):
        assert 'CORRDT 9,3000000,0,0,0,0,0,0' == mb.corrdt(9, 3000000, 0,
                                                           0, 0, 0, 0,
                                                           0).message

    def test_corrdt_query(self, mb):
        assert 'CORRDT? 9,1' == mb.corrdt_query(9, 1).message

    def test_corrl(self, mb):
        assert 'CORRL 9,3000000' == mb.corrl(9, 3000000).message

    def test_corrl_query(self, mb):
        assert 'CORRL? 9' == mb.corrl_query(9).message
        mb.clear_message_queue()
        assert 'CORRL? 9' == mb.corrl_query(9).message

    def test_corrser_query(self, mb: MessageBuilder):
        assert 'CORRSER? 101,1,1e-07,1e-08,10' == \
               mb.corrser_query(101,
                                True,
                                1E-7,
                                1E-8,
                                10).message

    def test_corrst(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_corrst_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_dcorr(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_dcorr_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_dcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_di(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_diag_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_do(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_dsmplarm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_dsmplflush(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_dsmplsetup(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_dv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_dz(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_emg_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_end(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ercmaa(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ercmaa_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ercmagrd(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ercmagrd_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ercmaio(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ercmaio_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ercmapfgd(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpa(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpa_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpe(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpe_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpl(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpl_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpp_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpqg(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpqg_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpr(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhpr_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhps(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhps_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhvca(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhvca_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhvctst_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhvp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhvp_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhvpv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhvs(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erhvs_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ermod(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ermod_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfda(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfda_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfdp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfdp_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfds(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfds_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfga(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfga_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfgp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfgp_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfgr(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfgr_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfqg(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfqg_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpftemp_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfuhca(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfuhca_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfuhccal_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfuhcmax_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erpfuhctst(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_err_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_errx_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ers_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erssp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_erssp_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_eruhva(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_eruhva_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_fc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_fl(self, mb):
        assert 'FL 1' == mb.fl(True).message
        mb.clear_message_queue()
        assert 'FL 0,1,3,5' == mb.fl(False, [1, 3, 5]).message
        mb.clear_message_queue()
        assert 'FL 0,1,2,3,4,5,6,7,8,9,10' == \
               mb.fl(False, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).message
        mb.clear_message_queue()
        with pytest.raises(ValueError):
            x = mb.fl(False, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).message

    def test_fmt(self, mb):
        assert 'FMT 1' == \
               mb.fmt(c.FMT.Format.ASCII_12_DIGITS_WITH_HEADER_CRLF_EOI
                      ).message
        mb.clear_message_queue()
        assert 'FMT 2,1' == \
               mb.fmt(c.FMT.Format.ASCII_12_DIGITS_NO_HEADER_CRLF_EOI,
                      c.FMT.Mode.PRIMARY_SOURCE_OUTPUT_DATA).message

    def test_hvsmuop(self, mb):
        self.skip()

    def test_hvsmuop_query(self, mb):
        assert 'HVSMUOP?' == \
               mb.hvsmuop_query().message

    def test_idn_query(self, mb):
        assert 'IN' == mb.in_().message
        mb.clear_message_queue()
        assert 'IN 1,2,3,5,6' == mb.in_([1, 2, 3, 5, 6]).message

    def test_imp(self, mb):
        assert 'IMP 10' == mb.imp(c.IMP.MeasurementMode.Z_THETA_RAD).message

    def test_in_(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_intlkvth(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_intlkvth_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lgi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lgv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lim(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lim_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lmn(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lop_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lrn_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lsi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lsm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lssi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lssv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lst_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lstm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lsv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_lsvm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mcc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mcpnt(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mcpnx(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mcpt(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mcpws(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mcpwnx(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ml(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mm(self, mb):
        assert 'MM 1,1' == \
               mb.mm(mode=c.MM.Mode.SPOT,
                     channels=[c.ChNr.SLOT_01_CH1]).message
        mb.clear_message_queue()
        assert 'MM 2,1,3' == mb.mm(mode=c.MM.Mode.STAIRCASE_SWEEP,
                                   channels=[c.ChNr.SLOT_01_CH1,
                                             c.ChNr.SLOT_03_CH1]).message

    def test_msc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_msp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mt(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mtdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_mv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_nub_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_odsw(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_odsw_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_opc_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_os(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_osx(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pa(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pad(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pax(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pch(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pch_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pt(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ptdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pwdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pwi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_pwv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_qsc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_qsl(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_qsm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_qso(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_qsr(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_qst(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_qsv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_qsz(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_rc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_rcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ri(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_rm(self, mb):
        assert 'RM 1,2' == mb.rm(1, c.RM.Mode.AUTO_UP).message
        mb.clear_message_queue()
        assert 'RM 2,3,60' == mb.rm(2, c.RM.Mode.AUTO_UP_DOWN, 60).message
        mb.clear_message_queue()
        with pytest.raises(ValueError):
            mb.rm(c.ChNr.SLOT_01_CH1, c.RM.Mode.DEFAULT, 22)

    def test_rst(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ru(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_rv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_rz(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sal(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sap(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sar(self, mb):
        assert 'SAR 1,0' == mb.sar(1,
                                   enable_picoamp_autoranging=True).message

    def test_scr(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ser(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ser_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sim(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sim_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sopc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sopc_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sovc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sovc_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spm_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spper(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spper_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sprm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sprm_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spst_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spt(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spt_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spupd(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_spv_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sre(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_sre_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_srp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ssl(self, mb):
        assert 'SSL 9,0' == mb.ssl(9, enable_indicator_led=False).message

    def test_ssp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ssr(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_st(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_stb_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_stgp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_stgp_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tacv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tdi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tdv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tgmo(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tgp(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tgpc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tgsi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tgso(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tgxo(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ti(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tiv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tmacv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tmdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tsc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tsq(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tsr(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tst(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ttc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tti(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ttiv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ttv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_tv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_unt_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_var(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_var_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wacv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wat(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wfc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wm(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wmacv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wmdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wmfc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wncc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wnu_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wnx(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_ws(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wsi(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wsv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wt(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wtacv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wtdcv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wtfc(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wv(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_wz_query(self, mb):
        # assert '' == mb.().message
        self.skip()

    def test_xe(self, mb):
        # assert '' == mb.().message
        self.skip()
