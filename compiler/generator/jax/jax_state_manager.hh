/************************************************************************
 ************************************************************************
    FAUST compiler - JAX State Manager
    Copyright (C) 2025 GRAME, Centre National de Creation Musicale
 ************************************************************************
 ************************************************************************/

#ifndef _JAX_STATE_MANAGER_H
#define _JAX_STATE_MANAGER_H

#include <set>
#include <string>
#include <vector>

/**
 * JAXStateManager - Manages JAX backend context state
 * 
 * Replaces multiple boolean flags with a proper state management system.
 * Uses RAII pattern for automatic context management.
 */
class JAXStateManager {
public:
    enum class Context {
        NONE,
        SETUP,
        TICK,
        STATIC_INIT,
        INLINE_SUBCONTAINER,
        INITIALIZE_CARRY
    };
    
    /**
     * RAII helper for automatic context management
     * Usage:
     *   {
     *     JAXStateManager::ContextScope scope(manager, Context::SETUP);
     *     // All code here has SETUP context
     *   } // Context automatically restored when scope ends
     */
    class ContextScope {
    private:
        JAXStateManager& fManager;
        
    public:
        ContextScope(JAXStateManager& manager, Context context) : fManager(manager) {
            fManager.pushContext(context);
        }
        
        ~ContextScope() {
            fManager.popContext();
        }
        
        // Prevent copying
        ContextScope(const ContextScope&) = delete;
        ContextScope& operator=(const ContextScope&) = delete;
    };
    
private:
    std::vector<Context> fContextStack;
    std::set<std::string> fInlineSubcontainerLocals;
    
public:
    JAXStateManager() {
        fContextStack.push_back(Context::NONE);
    }
    
    void pushContext(Context context) {
        fContextStack.push_back(context);
    }
    
    void popContext() {
        if (fContextStack.size() > 1) {
            fContextStack.pop_back();
        }
    }
    
    Context getCurrentContext() const {
        return fContextStack.back();
    }
    
    // Convenience methods that replace boolean flags
    bool inSetup() const {
        return getCurrentContext() == Context::SETUP;
    }
    
    bool inTick() const {
        return getCurrentContext() == Context::TICK;
    }
    
    bool inStaticInit() const {
        return getCurrentContext() == Context::STATIC_INIT;
    }
    
    bool inInlineSubcontainer() const {
        return getCurrentContext() == Context::INLINE_SUBCONTAINER;
    }
    
    bool inInitializeCarry() const {
        return getCurrentContext() == Context::INITIALIZE_CARRY;
    }
    
    // numpy is used in setup, static init, and initialize_carry contexts
    bool useNumpy() const {
        Context ctx = getCurrentContext();
        return ctx == Context::SETUP || ctx == Context::STATIC_INIT || 
               ctx == Context::INLINE_SUBCONTAINER || ctx == Context::INITIALIZE_CARRY;
    }
    
    // Inline subcontainer local variable management
    void addInlineSubcontainerLocal(const std::string& name) {
        fInlineSubcontainerLocals.insert(name);
    }
    
    bool isInlineSubcontainerLocal(const std::string& name) const {
        return fInlineSubcontainerLocals.find(name) != fInlineSubcontainerLocals.end();
    }
    
    void clearInlineSubcontainerLocals() {
        fInlineSubcontainerLocals.clear();
    }
    
    const std::set<std::string>& getInlineSubcontainerLocals() const {
        return fInlineSubcontainerLocals;
    }
    
    // Debug helpers
    std::string getContextString() const {
        switch (getCurrentContext()) {
            case Context::NONE: return "NONE";
            case Context::SETUP: return "SETUP";
            case Context::TICK: return "TICK";
            case Context::STATIC_INIT: return "STATIC_INIT";
            case Context::INLINE_SUBCONTAINER: return "INLINE_SUBCONTAINER";
            case Context::INITIALIZE_CARRY: return "INITIALIZE_CARRY";
            default: return "UNKNOWN";
        }
    }
    
    std::string getContextStackString() const {
        std::string result = "[";
        for (size_t i = 0; i < fContextStack.size(); ++i) {
            if (i > 0) result += " -> ";
            switch (fContextStack[i]) {
                case Context::NONE: result += "NONE"; break;
                case Context::SETUP: result += "SETUP"; break;
                case Context::TICK: result += "TICK"; break;
                case Context::STATIC_INIT: result += "STATIC_INIT"; break;
                case Context::INLINE_SUBCONTAINER: result += "INLINE_SUBCONTAINER"; break;
                case Context::INITIALIZE_CARRY: result += "INITIALIZE_CARRY"; break;
                default: result += "UNKNOWN"; break;
            }
        }
        result += "]";
        return result;
    }
};

#endif