/************************************************************************
 ************************************************************************
    FAUST compiler - JAX Variable Classifier
    Copyright (C) 2025 GRAME, Centre National de Creation Musicale
 ************************************************************************
 ************************************************************************/

#ifndef _JAX_VARIABLE_CLASSIFIER_H
#define _JAX_VARIABLE_CLASSIFIER_H

#include <string>
#include <regex>
#include <cctype>

/**
 * JAXVariableClassifier - Centralizes variable classification logic
 * 
 * This class encapsulates all the scattered variable name pattern matching
 * into a single, maintainable location.
 */
class JAXVariableClassifier {
public:
    enum class VarCategory {
        UI_PARAMETER,      // fButton*, fHslider*, fVslider*, fCheckbox*, fEntry*
        CONSTANT,          // fConst*, iConst*, fSampleRate
        STATIC_TABLE,      // ftbl0*SIG*, itbl0*SIG* (compile-time initialized)
        READ_WRITE_TABLE,  // ftbl* (non-ftbl0), itbl* (non-itbl0)
        WAVEFORM_DATA,     // f*Wave*, i*SIG* (but not _idx)
        INDEX_VARIABLE,    // *_idx
        DELAY_LINE,        // fRec*, fVec*, iRec*, iVec*
        TEMPORARY,         // fTemp*, iTemp*, fSlow*, iSlow*
        STATE_VARIABLE,    // IOTA, other state vars
        BARGRAPH,          // fVbargraph*, fHbargraph*
        SOUNDFILE,         // fSoundfile*
        OTHER              // Everything else
    };
    
private:
    std::string fClassName;
    
public:
    explicit JAXVariableClassifier(const std::string& className) : fClassName(className) {}
    
    /**
     * Main classification method
     */
    VarCategory classifyVariable(const std::string& name) const {
        // UI Parameters
        if (isUIParameter(name)) return VarCategory::UI_PARAMETER;
        
        // Index variables (check before other categories that might contain _idx)
        if (isIndexVariable(name)) return VarCategory::INDEX_VARIABLE;
        
        // Static tables (must check before waveform data)
        if (isStaticTable(name)) return VarCategory::STATIC_TABLE;
        
        // Read-write tables
        if (isReadWriteTable(name)) return VarCategory::READ_WRITE_TABLE;
        
        // Waveform data
        if (isWaveformData(name)) return VarCategory::WAVEFORM_DATA;
        
        // Constants
        if (isConstant(name)) return VarCategory::CONSTANT;
        
        // Delay lines
        if (isDelayLine(name)) return VarCategory::DELAY_LINE;
        
        // Temporary variables
        if (isTemporary(name)) return VarCategory::TEMPORARY;
        
        // Bargraphs
        if (isBargraph(name)) return VarCategory::BARGRAPH;
        
        // Soundfiles
        if (isSoundfile(name)) return VarCategory::SOUNDFILE;
        
        // Special state variables
        if (name == "IOTA") return VarCategory::STATE_VARIABLE;
        
        return VarCategory::OTHER;
    }
    
    // Individual classification methods
    
    bool isUIParameter(const std::string& name) const {
        return name.find("fButton") == 0 ||
               name.find("fHslider") == 0 ||
               name.find("fVslider") == 0 ||
               name.find("fCheckbox") == 0 ||
               name.find("fEntry") == 0;
    }
    
    bool isConstant(const std::string& name) const {
        return name.find("fConst") == 0 ||
               name.find("iConst") == 0 ||
               name == "fSampleRate";
    }
    
    bool isStaticTable(const std::string& name) const {
        // Static tables are ftbl0* or itbl0* that contain SIG
        return ((name.find("ftbl0") == 0 || name.find("itbl0") == 0) && 
                name.find("SIG") != std::string::npos);
    }
    
    bool isReadWriteTable(const std::string& name) const {
        // Read-write tables are ftbl* or itbl* but NOT ftbl0*/itbl0*
        return ((name.find("ftbl") == 0 && name.find("ftbl0") != 0) ||
                (name.find("itbl") == 0 && name.find("itbl0") != 0));
    }
    
    bool isWaveformData(const std::string& name) const {
        // Waveform data includes patterns like fmydspWave0, imydspSIG1
        // but excludes index variables (those ending with _idx)
        if (name.find("_idx") != std::string::npos) return false;
        
        // Check for Wave patterns
        if (name.find("Wave") != std::string::npos) {
            return name[0] == 'f' || name[0] == 'i';
        }
        
        // Check for SIG patterns that are waveform data (not table names)
        if (name.find("SIG") != std::string::npos && 
            (name[0] == 'f' || name[0] == 'i') &&
            name.find(fClassName) != std::string::npos &&
            !isStaticTable(name)) {
            return true;
        }
        
        return false;
    }
    
    bool isIndexVariable(const std::string& name) const {
        return name.find("_idx") != std::string::npos;
    }
    
    bool isDelayLine(const std::string& name) const {
        return name.find("fRec") == 0 ||
               name.find("fVec") == 0 ||
               name.find("iRec") == 0 ||
               name.find("iVec") == 0;
    }
    
    bool isTemporary(const std::string& name) const {
        return name.find("fTemp") == 0 ||
               name.find("iTemp") == 0 ||
               name.find("fSlow") == 0 ||
               name.find("iSlow") == 0;
    }
    
    bool isBargraph(const std::string& name) const {
        return name.find("fVbargraph") == 0 ||
               name.find("fHbargraph") == 0;
    }
    
    bool isSoundfile(const std::string& name) const {
        // Soundfile variables are like fSoundfile0, fSoundfile1, etc.
        // but NOT cache variables like fSoundfile0ca, fSoundfile0ca_le0, etc.
        if (name.find("fSoundfile") != 0) return false;
        
        // Check if it's just fSoundfileN (where N is a number)
        size_t pos = std::string("fSoundfile").length();
        while (pos < name.length() && std::isdigit(name[pos])) {
            pos++;
        }
        // If we've consumed the whole string, it's a soundfile
        // If there are more characters (like "ca"), it's not
        return pos == name.length();
    }
    
    // Helper methods for common queries
    
    bool isScalarDelay(const std::string& name, int size) const {
        return isDelayLine(name) && size <= 1;
    }
    
    bool isArrayDelay(const std::string& name, int size) const {
        return isDelayLine(name) && size > 1;
    }
    
    bool shouldBeInState(VarCategory category) const {
        switch (category) {
            case VarCategory::DELAY_LINE:
            case VarCategory::STATE_VARIABLE:
            case VarCategory::READ_WRITE_TABLE:
                return true;
            default:
                return false;
        }
    }
    
    bool shouldBeInstanceAttribute(VarCategory category) const {
        switch (category) {
            case VarCategory::CONSTANT:
            case VarCategory::STATIC_TABLE:
            case VarCategory::WAVEFORM_DATA:
            case VarCategory::SOUNDFILE:
                return true;
            default:
                return false;
        }
    }
    
    bool isAccessedInTick(VarCategory category) const {
        // Most variables are accessed in tick except pure setup-time variables
        switch (category) {
            case VarCategory::SOUNDFILE:
                return false;  // Soundfiles are loaded in setup
            default:
                return true;
        }
    }
    
    // Debug helper
    std::string getCategoryString(VarCategory category) const {
        switch (category) {
            case VarCategory::UI_PARAMETER: return "UI_PARAMETER";
            case VarCategory::CONSTANT: return "CONSTANT";
            case VarCategory::STATIC_TABLE: return "STATIC_TABLE";
            case VarCategory::READ_WRITE_TABLE: return "READ_WRITE_TABLE";
            case VarCategory::WAVEFORM_DATA: return "WAVEFORM_DATA";
            case VarCategory::INDEX_VARIABLE: return "INDEX_VARIABLE";
            case VarCategory::DELAY_LINE: return "DELAY_LINE";
            case VarCategory::TEMPORARY: return "TEMPORARY";
            case VarCategory::STATE_VARIABLE: return "STATE_VARIABLE";
            case VarCategory::BARGRAPH: return "BARGRAPH";
            case VarCategory::SOUNDFILE: return "SOUNDFILE";
            case VarCategory::OTHER: return "OTHER";
            default: return "UNKNOWN";
        }
    }
};

#endif