import 'package:chat2learnfe/model/dto/ReportDetailDTO.dart';
import 'package:chat2learnfe/services/ReportClient.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:chat2learnfe/widgets/CircularPercentInducator.dart';
import 'package:chat2learnfe/widgets/FadeAnimation.dart';
import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

class ChartData {
  ChartData(this.x, this.y, [this.color]);
  final String x;
  final double y;
  final Color? color;
}

class ReportPage extends StatefulWidget {
  const ReportPage({super.key});

  @override
  State<ReportPage> createState() => _ReportPageState();
}

class _ReportPageState extends State<ReportPage> {
  late ReportClient _reportClient;
  List<ChartData> chartData = [];
  List<ReportErrorCountDTOList> _reportErrorCountDTOList = [];
  double _errorRate = 0;
  double _score = 0;

  void _createClients() {
    String token = sharedPrefs.token;
    _reportClient = ReportClient(token);
  }

  void _getReport() async {
    try {
      ReportDetailDTO reportDetailDTO = await _reportClient.getReport();
      setState(() {
        chartData = reportDetailDTO.reportErrorCountDTOList!
            .map((e) => ChartData(e.code!, e.count!.toDouble()))
            .toList();
        _reportErrorCountDTOList = reportDetailDTO.reportErrorCountDTOList!;
        _errorRate = reportDetailDTO.messageCount! != 0
            ? ((reportDetailDTO.errorCount! / reportDetailDTO.messageCount!) *
                100)
            : 0;
        _score = reportDetailDTO.averageScore!;
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        backgroundColor: Colors.red,
        content: Text(e.toString()),
      ));
    }
  }

  @override
  void initState() {
    super.initState();
    _createClients();
    _getReport();
  }

  List<Widget> getErrorTypes(
      List<ReportErrorCountDTOList> reportErrorCountDTOList) {
    return reportErrorCountDTOList
        .map(
          (e) => FadeAnimation(
            2.0,
            ListTile(
                title: Text(e.code ?? ""),
                subtitle: Text(e.description ?? ""),
                trailing: Text(e.count.toString())),
          ),
        )
        .toList();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: ListView(children: [
            Container(
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Column(children: [
                    const SizedBox(height: 15),
                    const FadeAnimation(
                      1.2,
                      FittedBox(
                        fit: BoxFit.scaleDown,
                        child: Text(
                          "Percentage of grammer \nmistakes",
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 12),
                        ),
                      ),
                    ),
                    const SizedBox(height: 40),
                    FadeAnimation(
                      1.8,
                      Container(
                        child: CircularPercentIndicator(
                          radius: 40,
                          backgroundColor: Color.lerp(
                              Colors.green, Colors.red, _errorRate / 100)!,
                          center: Text("${_errorRate.toStringAsFixed(1)}%"),
                        ),
                      ),
                    ),
                  ]),
                  Column(children: [
                    const SizedBox(height: 15),
                    const FadeAnimation(
                      1.2,
                      Text(
                        "Avarage Score",
                        textAlign: TextAlign.center,
                        style: TextStyle(fontSize: 12),
                      ),
                    ),
                    const SizedBox(height: 40),
                    FadeAnimation(
                      1.8,
                      Container(
                        child: CircularPercentIndicator(
                          radius: 40,
                          backgroundColor:
                              Color.lerp(Colors.red, Colors.green, _score)!,
                          center: Text("${(_score * 100).toStringAsFixed(1)}%"),
                        ),
                      ),
                    ),
                  ]),
                ],
              ),
            ),
            FadeAnimation(
              2.0,
              Container(
                child: chartData.isEmpty
                    ? const Center(
                        child: Text("No data to display"),
                      )
                    : SfCircularChart(
                        title: ChartTitle(text: 'Most common mistake'),
                        legend: Legend(
                            position: LegendPosition.bottom,
                            isVisible: true,
                            overflowMode: LegendItemOverflowMode.wrap),
                        series: <CircularSeries>[
                          // Render pie chart
                          PieSeries<ChartData, String>(
                              dataSource: chartData,
                              enableTooltip: true,
                              dataLabelSettings:
                                  const DataLabelSettings(isVisible: true),
                              pointColorMapper: (ChartData data, _) =>
                                  data.color,
                              xValueMapper: (ChartData data, _) => data.x,
                              yValueMapper: (ChartData data, _) => data.y)
                        ],
                      ),
              ),
            ),
            getErrorTypes(_reportErrorCountDTOList).isEmpty
                ? const Center(
                    child: Text("No data to display"),
                  )
                : Column(
                    children: getErrorTypes(_reportErrorCountDTOList),
                  ),
          ]),
        ),
      ],
    );
  }
}
