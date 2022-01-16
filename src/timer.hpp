#pragma once

#include <string>
#include <cmath>
#include <chrono>
#include <fmt/core.h>
#include <fmt/color.h>


namespace timer {

using namespace std::chrono;

inline std::string to_string(double d) {
	std::string o = fmt::format("{:09.3f}",d);
	for (int i=0; i<o.length(); i++) if (o[i]=='0' and i<o.length()-2 and o[i+1]!='.') o[i]=' '; else break;
	return o;

	// This also looks decent
	std::string s = std::to_string(d);
	if (s.find("e") != std::string::npos) return s;
	auto dot = s.find(".");
	if (dot == std::string::npos) return s;
	for (int i=s.length()-1; i>0; i--) {
		if (s[i] != '0') return s.substr(0,i);
	}
	return "";
}

template <class T>
inline double getNanoDiff(const T& a, const T& b) {
	return duration_cast<nanoseconds>(b-a).count();
}
template <class T>
inline double getSecondDiff(const T& a, const T& b) {
	return duration_cast<nanoseconds>(b-a).count() * 1e-9;
}
inline std::string prettyPrintNanos(double d) {
	if (d > 1e9) return to_string(d*1e-9) + "s ";
	else if (d > 1e6) return to_string(d*1e-6) + "ms";
	else if (d > 1e3) return to_string(d*1e-3) + "μs";
	else return to_string(d) + "ns";
}
inline std::string prettyPrintSeconds(double d) {
	if (d < 1e-6) return to_string(d*1e9) + "ns";
	if (d < 1e-3) return to_string(d*1e6) + "μs";
	else if (d < 1) return to_string(d*1e3) + "ms";
	else return to_string(d) + "s ";
}

template <class T = high_resolution_clock>
struct SimpleTimerGuard {
	typename T::time_point st;
	const std::string& msg;
	inline SimpleTimerGuard(const std::string& msg) : msg(msg) {
		st = T::now();
	}
	inline ~SimpleTimerGuard() {
		double nanos = getNanoDiff(st, T::now());
		fmt::print(msg, prettyPrintNanos(nanos));
	}
};

struct Timer {
	std::string name;
	int n = 0;
	double acc1 = 0;
	double acc2 = 0;

	inline Timer(const std::string& name) : name(name) {
	}
	inline ~Timer() {
		double nd = n;
		double mu = acc1 / nd;
		double sd = std::sqrt(acc2/nd - mu*mu);
		//fmt::print(" - Task '{}' took {} (mu {}, std {})\n", name, prettyPrintSeconds(acc1), prettyPrintSeconds(mu), prettyPrintSeconds(sd));

		auto name_ = fmt::format(fmt::fg(fmt::color::pink), "{:>12s}", name);
		auto acc_ = fmt::format(fmt::fg(fmt::color::steel_blue), "{}", prettyPrintSeconds(acc1));
		auto mu_ = fmt::format(fmt::fg(fmt::color::orange), "{}", prettyPrintSeconds(mu));
		auto sd_ = fmt::format(fmt::fg(fmt::color::light_green), "{}", prettyPrintSeconds(sd));
		fmt::print(" - Task {} took {} (avg {} ± {})\n",
				name_,
				acc_,
				mu_,
				sd_);
	}

	void compose(const Timer& o) {
		n += o.n;
		acc1 += o.acc1;
		acc2 += o.acc2;
	}

	void pushMeasurement(double dt, int N=1) {
		n += N;
		acc1 += dt;
		acc2 += dt*dt;
	}
};
template <class T = high_resolution_clock>
struct TimerMeasurement {
	Timer& timer;
	typename T::time_point st;
	int N;
	TimerMeasurement(Timer& timer, int N=1) : timer(timer), st(T::now()), N(N) { }
	~TimerMeasurement() {
		auto et { T::now() };
		timer.pushMeasurement(getSecondDiff(st, et), N);
	}

};
}

using timer::TimerMeasurement;
using timer::Timer;
